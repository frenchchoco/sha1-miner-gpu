#include "pool_client.hpp"

#include <regex>

#include "../logging/logger.hpp"
#include "../tls/tls.hpp"
#include "gpu_platform.hpp"

namespace MiningPool {
    // ParsedUrl implementation
    PoolClient::ParsedUrl PoolClient::parse_url(const std::string &url)
    {
        ParsedUrl result;

        // Updated regex that properly handles localhost and other hostnames
        const std::regex url_regex(R"(^(wss?):\/\/([^:\/]+)(?::(\d+))?(\/.*)?$)");
        std::smatch matches;

        if (std::regex_match(url, matches, url_regex)) {
            result.is_secure = (matches[1].str() == "wss");
            result.host      = matches[2].str();
            // Extract port with proper defaults
            if (matches[3].length()) {
                result.port = matches[3].str();
            } else {
                result.port = result.is_secure ? "443" : "80";
            }
            result.path = matches[4].length() ? matches[4].str() : "/";
            // Debug output
            LOG_DEBUG("CLIENT", "Parsed URL - Host: ", result.host, ", Port: ", result.port, ", Path: ", result.path,
                      ", Secure: ", result.is_secure);
        } else {
            throw std::invalid_argument("Invalid WebSocket URL: " + url);
        }

        return result;
    }

    // PoolClient constructor
    PoolClient::PoolClient(const PoolConfig &config, IPoolEventHandler *handler)
        : config_(config), event_handler_(handler)
    {
        worker_stats_.worker_id       = config.worker_name;
        worker_stats_.connected_since = std::chrono::steady_clock::now();

        // Initialize work guard to keep io_context running
        work_guard_ = std::make_unique<net::executor_work_guard<net::io_context::executor_type>>(ioc_.get_executor());
    }

    // PoolClient destructor
    PoolClient::~PoolClient()
    {
        disconnect();
    }

    bool PoolClient::connect()
    {
        if (connected_.load()) {
            return true;
        }

        running_ = true;

        try {
            auto parsed = parse_url(config_.url);

            // Ensure we're using port 443 for wss://
            if (parsed.is_secure && parsed.port == "80") {
                parsed.port = "443";
            }

            // Start IO thread first
            io_thread_ = std::make_unique<std::thread>(&PoolClient::io_loop, this);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Start auxiliary threads
            message_processor_thread_ = std::thread(&PoolClient::message_processor_loop, this);
            keepalive_thread_         = std::thread(&PoolClient::keepalive_loop, this);

            LOG_INFO("CLIENT", "Connecting to ", parsed.host, ":", parsed.port, " (",
                     parsed.is_secure ? "secure" : "plain", ")");
            LOG_DEBUG("CLIENT", "WebSocket path: ", parsed.path);

            std::promise<bool> connection_promise;
            auto connection_future = connection_promise.get_future();

            net::post(ioc_, [this, parsed, &connection_promise]() {
                try {
                    // DNS resolution
                    tcp::resolver resolver{ioc_};
                    boost::system::error_code ec;

                    LOG_DEBUG("CLIENT", "Resolving ", parsed.host, ":", parsed.port);
                    auto const results = resolver.resolve(parsed.host, parsed.port, ec);

                    if (ec) {
                        throw std::runtime_error("DNS resolution failed: " + ec.message());
                    }

                    // Log resolved addresses
                    for (auto const &endpoint : results) {
                        LOG_DEBUG("CLIENT", "Resolved to: ", endpoint.endpoint().address().to_string(), ":",
                                  endpoint.endpoint().port());
                    }

                    if (config_.use_tls) {
                        // Create SSL context with Chrome fingerprint
                        ssl_ctx_ = ssl::context(ssl::context::tls_client);

                        // Apply Chrome configuration
                        ChromeTLSConfig::configureContext(ssl_ctx_);

                        // Certificate verification
                        if (config_.verify_server_cert) {
                            ssl_ctx_.set_verify_mode(ssl::verify_peer);
                            ssl_ctx_.set_default_verify_paths();

                            // More permissive Chrome-like verification
                            ssl_ctx_.set_verify_callback([](bool preverified, ssl::verify_context &ctx) {
                                if (!preverified) {
                                    X509_STORE_CTX *cts = ctx.native_handle();
                                    int error           = X509_STORE_CTX_get_error(cts);
                                    int depth           = X509_STORE_CTX_get_error_depth(cts);

                                    // Get certificate info
                                    X509 *cert = X509_STORE_CTX_get_current_cert(cts);
                                    if (cert) {
                                        char subject[256];
                                        X509_NAME_oneline(X509_get_subject_name(cert), subject, sizeof(subject));
                                        LOG_WARN("CLIENT", "Certificate verification issue at depth ", depth, ": ",
                                                 X509_verify_cert_error_string(error), " - Subject: ", subject);
                                    }

                                    // Chrome allows many certificate issues
                                    switch (error) {
                                        case X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY:
                                        case X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN:
                                        case X509_V_ERR_CERT_UNTRUSTED:
                                        case X509_V_ERR_UNABLE_TO_VERIFY_LEAF_SIGNATURE:
                                        case X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT:
                                        case X509_V_ERR_CERT_HAS_EXPIRED:
                                        case X509_V_ERR_CERT_NOT_YET_VALID:
                                            LOG_WARN("CLIENT", "Allowing certificate despite verification issue");
                                            return true;  // Accept anyway
                                        default:
                                            return false;  // Reject
                                    }
                                }
                                return preverified;
                            });
                        } else {
                            ssl_ctx_.set_verify_mode(ssl::verify_none);
                        }

                        // Load client certificates if specified
                        if (!config_.tls_cert_file.empty() && !config_.tls_key_file.empty()) {
                            ssl_ctx_.use_certificate_file(config_.tls_cert_file, ssl::context::pem);
                            ssl_ctx_.use_private_key_file(config_.tls_key_file, ssl::context::pem);
                        }

                        // Create WebSocket SSL stream
                        wss_ = std::make_unique<websocket::stream<ssl::stream<tcp::socket>>>(ioc_, ssl_ctx_);

                        // Get the lowest layer (TCP socket)
                        auto &socket = beast::get_lowest_layer(*wss_);

                        // Get the first endpoint to connect to
                        auto endpoint = results.begin()->endpoint();

                        // Open socket with the correct protocol based on the endpoint
                        if (endpoint.address().is_v4()) {
                            socket.open(tcp::v4(), ec);
                            LOG_DEBUG("CLIENT", "Opened IPv4 socket for address ", endpoint.address().to_string());
                        } else if (endpoint.address().is_v6()) {
                            socket.open(tcp::v6(), ec);
                            LOG_DEBUG("CLIENT", "Opened IPv6 socket for address ", endpoint.address().to_string());
                        } else {
                            throw std::runtime_error("Unknown address family");
                        }

                        if (ec) {
                            throw std::runtime_error("Failed to open socket: " + ec.message());
                        }

                        // NOW set socket options after the socket is open
                        socket.set_option(tcp::no_delay(true), ec);
                        if (ec) {
                            LOG_WARN("CLIENT", "Failed to set TCP_NODELAY: ", ec.message());
                            ec.clear();
                        }

                        socket.set_option(boost::asio::socket_base::keep_alive(true), ec);
                        if (ec) {
                            LOG_WARN("CLIENT", "Failed to set SO_KEEPALIVE: ", ec.message());
                            ec.clear();
                        }

                        socket.set_option(boost::asio::socket_base::reuse_address(true), ec);
                        if (ec) {
                            LOG_WARN("CLIENT", "Failed to set SO_REUSEADDR: ", ec.message());
                            ec.clear();
                        }

                        // Set buffer sizes
                        socket.set_option(boost::asio::socket_base::receive_buffer_size(65536), ec);
                        if (ec) {
                            LOG_WARN("CLIENT", "Failed to set receive buffer size: ", ec.message());
                            ec.clear();
                        }

                        socket.set_option(boost::asio::socket_base::send_buffer_size(65536), ec);
                        if (ec) {
                            LOG_WARN("CLIENT", "Failed to set send buffer size: ", ec.message());
                            ec.clear();
                        }

                        // Connect TCP
                        LOG_DEBUG("CLIENT", "Connecting TCP socket to ", endpoint.address().to_string(), ":",
                                  endpoint.port(), "...");
                        socket.connect(endpoint, ec);

                        if (ec) {
                            // If first address fails, try others
                            bool connected = false;
                            for (auto it = std::next(results.begin()); it != results.end(); ++it) {
                                auto alt_endpoint = it->endpoint();
                                LOG_DEBUG("CLIENT", "First connection failed, trying alternate address: ",
                                          alt_endpoint.address().to_string(), ":", alt_endpoint.port());

                                // Close and reopen socket with correct protocol
                                socket.close(ec);

                                if (alt_endpoint.address().is_v4()) {
                                    socket.open(tcp::v4(), ec);
                                } else if (alt_endpoint.address().is_v6()) {
                                    socket.open(tcp::v6(), ec);
                                }

                                if (!ec) {
                                    // Re-apply socket options
                                    socket.set_option(tcp::no_delay(true), ec);
                                    socket.set_option(boost::asio::socket_base::keep_alive(true), ec);
                                    socket.set_option(boost::asio::socket_base::reuse_address(true), ec);

                                    socket.connect(alt_endpoint, ec);
                                    if (!ec) {
                                        connected = true;
                                        LOG_INFO("CLIENT", "Connected to alternate address");
                                        break;
                                    }
                                }
                            }

                            if (!connected) {
                                throw std::runtime_error("TCP connect failed to all addresses: " + ec.message());
                            }
                        }

                        // Configure SSL before handshake
                        auto &ssl_stream = wss_->next_layer();
                        SSL *ssl         = ssl_stream.native_handle();
                        ChromeTLSConfig::configureSSLStream(ssl, parsed.host);

                        // Perform SSL handshake
                        LOG_DEBUG("CLIENT", "Starting SSL handshake with ", parsed.host, "...");
                        ssl_stream.handshake(ssl::stream_base::client, ec);

                        if (ec) {
                            // Detailed error reporting
                            std::string error_details;
                            char err_buf[256];
                            unsigned long ssl_err;

                            while ((ssl_err = ERR_get_error()) != 0) {
                                ERR_error_string_n(ssl_err, err_buf, sizeof(err_buf));
                                error_details += std::string(err_buf) + "; ";
                            }

                            throw std::runtime_error("SSL handshake failed: " + ec.message() +
                                                     " - OpenSSL: " + error_details);
                        }

                        // Log connection info
                        LOG_INFO("CLIENT", Color::GREEN, "SSL connected!", Color::RESET);
                        LOG_INFO("CLIENT", "  Protocol: ", SSL_get_version(ssl));
                        LOG_INFO("CLIENT", "  Cipher: ", SSL_get_cipher_name(ssl));

                        // Check ALPN
                        const unsigned char *alpn_proto = nullptr;
                        unsigned int alpn_proto_len     = 0;
                        SSL_get0_alpn_selected(ssl, &alpn_proto, &alpn_proto_len);
                        if (alpn_proto && alpn_proto_len > 0) {
                            std::string alpn(reinterpret_cast<const char *>(alpn_proto), alpn_proto_len);
                            LOG_INFO("CLIENT", "  ALPN: ", alpn);
                        }

                        // Configure WebSocket options BEFORE decorator
                        wss_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));

                        // Set WebSocket decorator with proper headers to fix "bad version"
                        wss_->set_option(websocket::stream_base::decorator([parsed](websocket::request_type &req) {
                            req.set(http::field::host, parsed.host);
                            req.set(http::field::upgrade, "websocket");
                            req.set(http::field::connection, "Upgrade");

                            // Chrome User-Agent
                            req.set(http::field::user_agent,
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36");

                            // Standard Chrome headers
                            req.set(http::field::accept_language, "en-US,en;q=0.9");
                            req.set(http::field::cache_control, "no-cache");
                            req.set(http::field::pragma, "no-cache");
                            req.set(http::field::origin, "https://" + parsed.host);

                            // WebSocket version (required)
                            req.set("Sec-WebSocket-Version", "13");

                            // Chrome Sec-Fetch headers
                            req.set("Sec-Fetch-Dest", "websocket");
                            req.set("Sec-Fetch-Mode", "websocket");
                            req.set("Sec-Fetch-Site", "same-origin");
                        }));

                        // WebSocket handshake
                        LOG_DEBUG("CLIENT", "Starting WebSocket handshake...");
                        LOG_DEBUG("CLIENT", "  Host: ", parsed.host);
                        LOG_DEBUG("CLIENT", "  Path: ", parsed.path);

                        wss_->handshake(parsed.host, parsed.path, ec);

                        if (ec) {
                            // Enhanced error handling for WebSocket
                            if (ec == websocket::error::upgrade_declined) {
                                LOG_ERROR("CLIENT", "WebSocket upgrade declined by server");
                                LOG_ERROR("CLIENT", "Check if the path '", parsed.path, "' supports WebSocket");
                                LOG_ERROR("CLIENT", "The server might expect a different path or protocol");
                            } else if (ec == websocket::error::bad_http_version) {
                                LOG_ERROR("CLIENT", "Bad HTTP version - server might not support WebSocket");
                            } else if (ec.message().find("bad version") != std::string::npos) {
                                LOG_ERROR("CLIENT", "WebSocket version mismatch");
                                LOG_ERROR("CLIENT", "Try removing the permessage-deflate extension");
                            }

                            throw std::runtime_error("WebSocket handshake failed: " + ec.message() +
                                                     " (category: " + ec.category().name() +
                                                     ", value: " + std::to_string(ec.value()) + ")");
                        }
                    } else {
                        // Plain WebSocket (non-TLS)
                        ws_ = std::make_unique<websocket::stream<tcp::socket>>(ioc_);

                        auto &socket = beast::get_lowest_layer(*ws_);

                        // Get the first endpoint
                        auto endpoint = results.begin()->endpoint();

                        // Open socket with correct protocol
                        if (endpoint.address().is_v4()) {
                            socket.open(tcp::v4(), ec);
                            LOG_DEBUG("CLIENT", "Opened IPv4 socket for address ", endpoint.address().to_string());
                        } else if (endpoint.address().is_v6()) {
                            socket.open(tcp::v6(), ec);
                            LOG_DEBUG("CLIENT", "Opened IPv6 socket for address ", endpoint.address().to_string());
                        } else {
                            throw std::runtime_error("Unknown address family");
                        }

                        if (ec) {
                            throw std::runtime_error("Failed to open socket: " + ec.message());
                        }

                        // Set options after socket is open
                        socket.set_option(tcp::no_delay(true), ec);
                        socket.set_option(boost::asio::socket_base::keep_alive(true), ec);

                        // Connect
                        socket.connect(endpoint, ec);
                        if (ec) {
                            // Try alternate addresses if available
                            bool connected = false;
                            for (auto it = std::next(results.begin()); it != results.end(); ++it) {
                                auto alt_endpoint = it->endpoint();
                                LOG_DEBUG("CLIENT", "First connection failed, trying alternate address: ",
                                          alt_endpoint.address().to_string(), ":", alt_endpoint.port());

                                socket.close(ec);

                                if (alt_endpoint.address().is_v4()) {
                                    socket.open(tcp::v4(), ec);
                                } else if (alt_endpoint.address().is_v6()) {
                                    socket.open(tcp::v6(), ec);
                                }

                                if (!ec) {
                                    socket.set_option(tcp::no_delay(true), ec);
                                    socket.set_option(boost::asio::socket_base::keep_alive(true), ec);

                                    socket.connect(alt_endpoint, ec);
                                    if (!ec) {
                                        connected = true;
                                        LOG_INFO("CLIENT", "Connected to alternate address");
                                        break;
                                    }
                                }
                            }

                            if (!connected) {
                                throw std::runtime_error("TCP connect failed to all addresses: " + ec.message());
                            }
                        }

                        ws_->set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));
                        ws_->set_option(websocket::stream_base::decorator([parsed](websocket::request_type &req) {
                            req.set(http::field::host, parsed.host + ":" + parsed.port);
                            req.set(http::field::upgrade, "websocket");
                            req.set(http::field::connection, "Upgrade");
                            req.set(http::field::user_agent, "SHA1-Miner/1.0 Boost.Beast");
                            req.set("Sec-WebSocket-Version", "13");
                        }));

                        ws_->handshake(parsed.host + ":" + parsed.port, parsed.path, ec);
                        if (ec) {
                            throw std::runtime_error("WebSocket handshake failed: " + ec.message());
                        }
                    }

                    connected_                    = true;
                    worker_stats_.connected_since = std::chrono::steady_clock::now();

                    LOG_INFO("CLIENT", Color::GREEN, "WebSocket connected successfully!", Color::RESET);

                    // Start reading
                    do_read();
                    connection_promise.set_value(true);
                } catch (const std::exception &e) {
                    LOG_ERROR("CLIENT", "Connection failed: ", e.what());
                    connection_promise.set_value(false);
                }
            });

            // Wait for connection result
            if (!connection_future.get()) {
                // running_ = false;
                connected_ = false;
                return false;
            }

            // Notify handler
            event_handler_->on_connected();

            // Send initial HELLO message
            HelloMessage hello;
            hello.protocol_version = PROTOCOL_VERSION;
            hello.client_version   = "SHA1-Miner/1.0";
            hello.capabilities     = {"gpu", "multi-gpu", "vardiff"};
            hello.user_agent       = "SHA1-Miner";

            Message msg;
            msg.type      = MessageType::HELLO;
            msg.id        = Utils::generate_message_id();
            msg.timestamp = Utils::current_timestamp_ms();
            msg.payload   = hello.to_json();

            LOG_INFO("CLIENT", "Sending HELLO message");
            send_message(msg);

            return true;
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Connection error: ", e.what());
            // running_ = false;
            connected_ = false;
            return false;
        }
    }

    void PoolClient::disconnect()
    {
        if (!running_.load()) {
            return;
        }

        running_       = false;
        connected_     = false;
        authenticated_ = false;

        // Close WebSocket connection
        try {
            if (config_.use_tls && wss_) {
                if (wss_->is_open()) {
                    wss_->close(websocket::close_code::normal);
                }
            } else if (ws_) {
                if (ws_->is_open()) {
                    ws_->close(websocket::close_code::normal);
                }
            }
        } catch (const std::exception &e) {
            // Ignore errors during shutdown
        }

        // Stop io_context
        if (work_guard_) {
            work_guard_.reset();
        }
        ioc_.stop();

        // Notify condition variables
        outgoing_cv_.notify_all();
        incoming_cv_.notify_all();

        // IMPORTANT: Don't join threads if we're being called from one of them
        // Check if we're being called from io_thread
        if (io_thread_ && io_thread_->joinable() && io_thread_->get_id() != std::this_thread::get_id()) {
            io_thread_->join();
        }
        // Check if we're being called from message_processor_thread
        if (message_processor_thread_.joinable() && message_processor_thread_.get_id() != std::this_thread::get_id()) {
            message_processor_thread_.join();
        }

        // Check if we're being called from keepalive_thread
        if (keepalive_thread_.joinable() && keepalive_thread_.get_id() != std::this_thread::get_id()) {
            keepalive_thread_.join();
        }
    }

    void PoolClient::io_loop()
    {
        try {
            ioc_.run();
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "IO loop error: ", e.what());
        }
    }

    void PoolClient::shutdown_connection()
    {
        LOG_DEBUG("CLIENT", "Shutting down connection...");
        // First, cancel all async operations
        if (config_.use_tls && wss_) {
            try {
                auto &socket = beast::get_lowest_layer(*wss_);
                boost::system::error_code ec;
                socket.cancel(ec);
                if (wss_->is_open()) {
                    wss_->async_close(websocket::close_code::normal, [](beast::error_code) {
                        // Ignore close errors during shutdown
                    });
                }
            } catch (...) {}
        } else if (ws_) {
            try {
                auto &socket = beast::get_lowest_layer(*ws_);
                boost::system::error_code ec;
                socket.cancel(ec);
                if (ws_->is_open()) {
                    ws_->async_close(websocket::close_code::normal, [](beast::error_code) {
                        // Ignore close errors during shutdown
                    });
                }
            } catch (...) {}
        }

        // Give async operations time to complete
        ioc_.run_for(std::chrono::milliseconds(100));

        // Now stop the io_context
        ioc_.stop();
    }

    void PoolClient::send_message(const Message &msg)
    {
        LOG_DEBUG("CLIENT", "send_message called for type: ", static_cast<int>(msg.type), ", ID: ", msg.id);
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "Cannot send message - not connected");
            return;
        }

        try {
            std::string payload = msg.serialize();
            LOG_DEBUG("CLIENT", "Message serialized, length: ", payload.length());
            LOG_TRACE("CLIENT", "Payload: ", payload);

            // Track pending requests if expecting response
            if (msg.type == MessageType::AUTH || msg.type == MessageType::SUBMIT_SHARE ||
                msg.type == MessageType::GET_JOB) {
                std::lock_guard lock(pending_mutex_);
                PendingRequest req;
                req.timestamp             = std::chrono::steady_clock::now();
                req.type                  = msg.type;
                pending_requests_[msg.id] = req;
                LOG_DEBUG("CLIENT", "Added message ID ", msg.id, " (type ", static_cast<int>(msg.type),
                          ") to pending requests");
            }

            // Queue message for sending
            {
                std::lock_guard lock(outgoing_mutex_);
                outgoing_queue_.push(msg);
                LOG_DEBUG("CLIENT", "Message queued, queue size: ", outgoing_queue_.size());
                outgoing_cv_.notify_one();
            }

            // Trigger write if not already writing
            net::post(ioc_, [this]() {
                LOG_DEBUG("CLIENT", "Posted do_write to io_context");
                do_write();
            });
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Exception in send_message: ", e.what());
        }
    }

    void PoolClient::do_read()
    {
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "do_read called but not connected!");
            return;
        }

        LOG_TRACE("CLIENT", "Setting up async_read");
        // Clear buffer before reading
        buffer_.consume(buffer_.size());
        auto read_handler = [this](beast::error_code ec, std::size_t bytes_transferred) {
            LOG_TRACE("CLIENT", "async_read completed - ec: ", ec, ", bytes: ", bytes_transferred);
            if (!ec) {
                LOG_TRACE("CLIENT", "Read data: ", beast::buffers_to_string(buffer_.data()));
            }
            on_read(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_read(buffer_, read_handler);
        } else if (ws_) {
            ws_->async_read(buffer_, read_handler);
        } else {
            LOG_ERROR("CLIENT", "ERROR: No websocket stream available!");
        }
    }

    void PoolClient::do_write()
    {
        if (!connected_.load() || write_in_progress_.load()) {
            return;
        }
        std::unique_lock lock(outgoing_mutex_);
        if (outgoing_queue_.empty()) {
            return;
        }
        write_in_progress_ = true;
        Message msg        = outgoing_queue_.front();
        outgoing_queue_.pop();
        lock.unlock();

        current_write_payload_ = msg.serialize();
        auto write_handler     = [this](const beast::error_code &ec, const std::size_t bytes_transferred) {
            write_in_progress_ = false;
            on_write(ec, bytes_transferred);
        };

        if (config_.use_tls && wss_) {
            wss_->async_write(net::buffer(current_write_payload_), write_handler);
        } else if (ws_) {
            ws_->async_write(net::buffer(current_write_payload_), write_handler);
        }
    }

    void PoolClient::on_write(beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec) {
            LOG_ERROR("CLIENT", "Write error: ", ec.message());
            return;
        }

        // Check if more messages to send
        std::lock_guard lock(outgoing_mutex_);
        if (!outgoing_queue_.empty()) {
            net::post(ioc_, [this]() { do_write(); });
        }
    }

    void PoolClient::on_read(beast::error_code ec, std::size_t bytes_transferred)
    {
        if (ec) {
            if (ec != websocket::error::closed) {
                std::string error_msg = ec.message();
#ifdef _WIN32
                if (error_msg.find('\xd9') != std::string::npos || error_msg.find('\xda') != std::string::npos ||
                    error_msg.find('\xc6') != std::string::npos) {
                    error_msg = "Connection closed by remote host (error code: " + std::to_string(ec.value()) + ")";
                }
#endif
                LOG_ERROR("CLIENT", "Read error: ", error_msg);
            }

            connected_     = false;
            authenticated_ = false;
            event_handler_->on_disconnected(ec.message());

            bool should_reconnect = running_.load() && (config_.reconnect_attempts < 0 ||  // -1 means infinite
                                                        reconnect_attempt_count_.load() <
                                                            static_cast<unsigned>(config_.reconnect_attempts));

            if (should_reconnect) {
                LOG_WARN(
                    "CLIENT", "Scheduling reconnect (attempt ", reconnect_attempt_count_.load() + 1,
                    config_.reconnect_attempts < 0 ? "/infinite" : "/" + std::to_string(config_.reconnect_attempts),
                    ")");
                // Schedule reconnect
                std::thread([this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    reconnect();
                }).detach();
            } else {
                LOG_ERROR("CLIENT", "Not reconnecting - running=", running_.load(),
                          ", attempts=", reconnect_attempt_count_.load(), "/", config_.reconnect_attempts);
            }
            return;
        }

        // Check if we're still connected before processing
        if (!connected_.load()) {
            LOG_ERROR("CLIENT", "Received data but not connected, ignoring");
            return;
        }

        std::string data = beast::buffers_to_string(buffer_.data());
        buffer_.consume(bytes_transferred);

        LOG_DEBUG("CLIENT", "Received ", bytes_transferred, " bytes");
        auto parsed = Message::deserialize(data);
        if (parsed) {
            LOG_DEBUG("CLIENT", "Successfully parsed message type: ", static_cast<int>(parsed->type),
                      ", id: ", parsed->id);

            // Check for error messages that should stop further reading
            if (parsed->type == MessageType::ERROR_PROBLEM) {
                int error_code = parsed->payload.value("code", 0);
                if (error_code == static_cast<int>(ErrorCode::INVALID_MESSAGE) ||
                    error_code == static_cast<int>(ErrorCode::PROTOCOL_ERROR)) {
                    LOG_ERROR("CLIENT", "Received fatal error, stopping reads");
                    connected_ = false;
                }
            }

            std::lock_guard lock(incoming_mutex_);
            incoming_queue_.push(*parsed);
            incoming_cv_.notify_one();
        }

        // Continue reading only if still connected
        if (connected_.load()) {
            do_read();
        }
    }

    bool PoolClient::is_authenticated() const
    {
        return authenticated_.load();
    }

    void PoolClient::keepalive_loop()
    {
        while (running_.load()) {
            for (int i = 0; i < config_.keepalive_interval_s && running_.load(); i++) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            if (!running_.load())
                break;

            if (connected_.load() && authenticated_.load()) {
                Message keepalive;
                keepalive.type      = MessageType::KEEPALIVE;
                keepalive.id        = Utils::generate_message_id();
                keepalive.timestamp = Utils::current_timestamp_ms();
                send_message(keepalive);
            }

            // Check for timed out requests
            check_pending_timeouts();
        }
        LOG_DEBUG("CLIENT", "Keepalive loop exiting");
    }

    void PoolClient::message_processor_loop()
    {
        LOG_DEBUG("CLIENT", "Message processor thread started (tid: ", std::this_thread::get_id(), ")");
        while (running_.load()) {
            std::unique_lock lock(incoming_mutex_);
            LOG_TRACE("CLIENT", "Message processor waiting for messages...");
            // Wait for messages with timeout to check running status
            bool has_message = incoming_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this] {
                bool result = !incoming_queue_.empty() || !running_.load();
                if (result) {
                    LOG_TRACE("CLIENT", "Wait condition met - queue empty: ", incoming_queue_.empty(),
                              ", running: ", running_.load());
                }
                return result;
            });

            if (!has_message) {
                LOG_TRACE("CLIENT", "Message processor timeout - no messages");
                continue;
            }

            while (!incoming_queue_.empty()) {
                Message msg = incoming_queue_.front();
                incoming_queue_.pop();
                LOG_DEBUG("CLIENT", "Dequeued message for processing - type: ", static_cast<int>(msg.type),
                          ", id: ", msg.id);
                lock.unlock();
                try {
                    process_message(msg);
                } catch (const std::exception &e) {
                    LOG_ERROR("CLIENT", "Exception processing message: ", e.what());
                }

                lock.lock();
            }
        }

        LOG_DEBUG("CLIENT", "Message processor thread stopped");
    }

    void PoolClient::process_message(const Message &msg)
    {
        // Remove from pending if this is a response
        {
            std::lock_guard lock(pending_mutex_);
            pending_requests_.erase(msg.id);
        }

        LOG_TRACE("CLIENT", "process_message called with type: ", static_cast<int>(msg.type), " (0x", std::hex,
                  static_cast<int>(msg.type), std::dec, ")");

        try {
            switch (msg.type) {
                case MessageType::WELCOME:
                    handle_welcome(WelcomeMessage::from_json(msg.payload));
                    break;
                case MessageType::AUTH_RESPONSE:
                    handle_auth_response(AuthResponseMessage::from_json(msg.payload));
                    break;
                case MessageType::NEW_JOB:
                    handle_new_job(JobMessage::from_json(msg.payload));
                    break;
                case MessageType::SHARE_RESULT:
                    handle_share_result(ShareResultMessage::from_json(msg.payload));
                    break;
                case MessageType::DIFFICULTY_ADJUST:
                    handle_difficulty_adjust(DifficultyAdjustMessage::from_json(msg.payload));
                    break;
                case MessageType::POOL_STATUS:
                    handle_pool_status(PoolStatusMessage::from_json(msg.payload));
                    break;
                case MessageType::ERROR_PROBLEM:  // This is 0x17 = 23
                    handle_error(msg);
                    break;
                case MessageType::KEEP_ALIVE_RESPONSE:
                    LOG_DEBUG("CLIENT", "Received KEEP_ALIVE_RESPONSE, ignoring");
                    break;
                case MessageType::REPORT_RECEIVED:
                    LOG_DEBUG("CLIENT", "Received REPORT_RECEIVED, ignoring");
                    break;
                case MessageType::RECONNECT:
                    // Server requested reconnect
                    reconnect();
                    break;
                default:
                    LOG_ERROR("CLIENT", "Unknown message type: ", static_cast<int>(msg.type), " (hex: 0x", std::hex,
                              static_cast<int>(msg.type), std::dec, ")");
            }
        } catch (const std::exception &e) {
            LOG_ERROR("CLIENT", "Error processing message type ", static_cast<int>(msg.type), ": ", e.what());
            event_handler_->on_error(ErrorCode::PROTOCOL_ERROR, e.what());
        }
    }

    void PoolClient::handle_welcome(const WelcomeMessage &welcome)
    {
        LOG_INFO("CLIENT", Color::GREEN, "Connected to pool: ", Color::RESET, welcome.pool_name, " v",
                 welcome.pool_version);

        // Check protocol compatibility
        if (welcome.protocol_version != PROTOCOL_VERSION) {
            LOG_WARN("CLIENT", "Protocol version mismatch. Pool: ", welcome.protocol_version,
                     ", Client: ", PROTOCOL_VERSION);
        }

        // Log minimum difficulty in bits
        LOG_INFO("CLIENT", "Minimum difficulty: ", welcome.min_difficulty, " bits");

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            worker_stats_.current_difficulty = welcome.min_difficulty;
        }

        // Proceed to authentication
        authenticate();
    }

    bool PoolClient::authenticate()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        AuthMessage auth;
        auth.method     = config_.auth_method;
        auth.username   = config_.username + "." + config_.worker_name;
        auth.password   = config_.password;
        auth.session_id = session_id_;

        // Add client info for better server-side difficulty adjustment
        ClientInfo client_info;

        // Estimate hashrate based on GPU count
        // The actual mining system will be initialized later by PoolMiningSystem
        int gpu_count = 0;
        gpuGetDeviceCount(&gpu_count);

        if (gpu_count > 0) {
            client_info.gpu_count = gpu_count;
            // Estimate based on typical GPU performance
            // Modern GPUs typically achieve 2-4 GH/s for SHA-1
            client_info.estimated_hashrate = gpu_count * 2.5e9;  // 2.5 GH/s per GPU
        } else {
            // CPU fallback (though not recommended)
            client_info.gpu_count          = 0;
            client_info.estimated_hashrate = 100e6;  // 100 MH/s for CPU
        }

        client_info.miner_version = "SHA1-Miner/1.0.0";
        auth.client_info          = client_info;

        LOG_INFO("CLIENT", "Sending AUTH message:");
        LOG_DEBUG("CLIENT", "  Method: ",
                  (auth.method == AuthMethod::WORKER_PASS ? "worker_pass"
                   : auth.method == AuthMethod::API_KEY   ? "api_key"
                                                          : "certificate"));
        LOG_DEBUG("CLIENT", "  Username: ", auth.username);
        LOG_DEBUG("CLIENT", "  Password: ", (auth.password.empty() ? "<empty>" : "<set>"));
        LOG_DEBUG("CLIENT", "  Estimated hashrate: ", client_info.estimated_hashrate / 1e9, " GH/s");
        LOG_DEBUG("CLIENT", "  GPU count: ", client_info.gpu_count);

        Message msg;
        msg.type      = MessageType::AUTH;
        msg.id        = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload   = auth.to_json();

        LOG_TRACE("CLIENT", "AUTH payload: ", msg.payload.dump());

        send_message(msg);
        return true;
    }

    void PoolClient::handle_auth_response(const AuthResponseMessage &response)
    {
        if (response.success) {
            authenticated_ = true;
            session_id_    = response.session_id;
            worker_id_     = response.worker_id;
            {
                std::lock_guard lock(stats_mutex_);
                worker_stats_.worker_id = worker_id_;

                // NEW: Set initial difficulty if provided
                if (response.initial_difficulty > 0) {
                    worker_stats_.current_difficulty = response.initial_difficulty;
                    LOG_INFO("CLIENT", "Initial difficulty set to ", response.initial_difficulty, " bits");
                }
            }

            event_handler_->on_authenticated(worker_id_);

            // Request initial job
            request_job();
        } else {
            authenticated_ = false;
            event_handler_->on_auth_failed(response.error_code, response.error_message);
        }
    }

    void PoolClient::request_job()
    {
        Message msg;
        msg.type      = MessageType::GET_JOB;
        msg.id        = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        send_message(msg);
    }

    void PoolClient::handle_new_job(const JobMessage &job)
    {
        PoolJob pool_job;
        pool_job.job_id        = job.job_id;
        pool_job.job_data      = job;
        pool_job.received_time = std::chrono::steady_clock::now();
        pool_job.expiry_time   = pool_job.received_time + std::chrono::seconds(job.expires_in_seconds);
        pool_job.is_active     = true;
        {
            std::lock_guard lock(jobs_mutex_);

            // Clean jobs if requested
            if (job.clean_jobs) {
                for (auto &[id, existing_job] : active_jobs_) {
                    existing_job.is_active = false;
                    event_handler_->on_job_cancelled(id);
                }
                active_jobs_.clear();
            }

            // Add new job
            active_jobs_[job.job_id] = pool_job;
            current_job_id_          = job.job_id;

            // Clean up expired jobs
            cleanup_expired_jobs();
        }

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            worker_stats_.last_job_time = std::chrono::steady_clock::now();
        }

        event_handler_->on_new_job(pool_job);
    }

    void PoolClient::submit_share(const Share &share)
    {
        if (!authenticated_.load()) {
            LOG_ERROR("SHARE", "Not authenticated, cannot submit share");
            return;
        }

        if (!connected_.load()) {
            LOG_ERROR("SHARE", "Not connected, cannot submit share");
            return;
        }

        // Validate share data
        if (share.job_id.empty()) {
            LOG_ERROR("SHARE", "Empty job ID, cannot submit share");
            return;
        }

        if (share.hash.empty()) {
            LOG_ERROR("SHARE", "Empty hash, cannot submit share");
            return;
        }

        // Validate hash format (should be 40 hex characters)
        if (share.hash.length() != 40) {
            LOG_ERROR("SHARE", "Invalid hash length: ", share.hash.length(), " (expected 40)");
            return;
        }

        // Check if hash contains only hex characters
        for (char c : share.hash) {
            if (!std::isxdigit(c)) {
                LOG_ERROR("SHARE", "Invalid hash character: '", c, "' in hash: ", share.hash);
                return;
            }
        }

        try {
            SubmitShareMessage submit;
            submit.job_id        = share.job_id;
            submit.nonce         = share.nonce;
            submit.hash          = share.hash;
            submit.matching_bits = share.matching_bits;
            submit.worker_name   = config_.worker_name;

            Message msg;
            msg.type      = MessageType::SUBMIT_SHARE;
            msg.id        = Utils::generate_message_id();
            msg.timestamp = Utils::current_timestamp_ms();
            msg.payload   = submit.to_json();

            // Track submission time for latency stats
            {
                std::lock_guard lock(stats_mutex_);
                worker_stats_.last_share_time = std::chrono::steady_clock::now();
            }

            send_message(msg);
        } catch (const std::exception &e) {
            LOG_ERROR("SHARE", "Exception in submit_share: ", e.what());
        }
    }

    void PoolClient::handle_share_result(const ShareResultMessage &result)
    {
        update_stats(result);

        if (result.status == ShareStatus::ACCEPTED) {
            event_handler_->on_share_accepted(result);
        } else {
            event_handler_->on_share_rejected(result);

            // Log rejection reason
            std::string reason = "Unknown";
            switch (result.status) {
                case ShareStatus::REJECTED_LOW_DIFFICULTY:
                    reason = "Low difficulty (insufficient bits)";
                    break;
                case ShareStatus::REJECTED_INVALID:
                    reason = "Invalid share";
                    break;
                case ShareStatus::REJECTED_STALE:
                    reason = "Stale share";
                    break;
                case ShareStatus::REJECTED_DUPLICATE:
                    reason = "Duplicate share";
                    break;
                default:
                    break;
            }

            LOG_WARN("SHARE", Color::RED, "Share rejected: ", reason, Color::RESET);
        }
    }

    void PoolClient::update_stats(const ShareResultMessage &result)
    {
        std::lock_guard lock(stats_mutex_);

        if (result.status == ShareStatus::ACCEPTED) {
            worker_stats_.shares_accepted++;
            worker_stats_.total_difficulty_accepted += result.difficulty_credited;
        } else {
            worker_stats_.shares_rejected++;
        }

        worker_stats_.last_share_time = std::chrono::steady_clock::now();
    }

    void PoolClient::handle_difficulty_adjust(const DifficultyAdjustMessage &adjust)
    {
        {
            std::lock_guard lock(stats_mutex_);
            worker_stats_.current_difficulty = adjust.new_difficulty;  // NOW IN BITS
        }

        LOG_INFO("POOL", "Difficulty adjusted to ", Color::BRIGHT_MAGENTA, adjust.new_difficulty, " bits", Color::RESET,
                 " (", adjust.reason, ")");

        // Log scaled difficulty for clarity
        double scaled = DifficultyConverter::bitsToScaledDifficulty(adjust.new_difficulty);
        LOG_DEBUG("POOL", "  Scaled difficulty: ", DifficultyConverter::formatDifficulty(scaled));

        event_handler_->on_difficulty_changed(adjust.new_difficulty);
    }

    void PoolClient::handle_pool_status(const PoolStatusMessage &status)
    {
        // Log epoch information with bit difficulty if available
        if (!status.extra_info.is_null() && status.extra_info.contains("epoch_info")) {
            auto epoch_info = status.extra_info["epoch_info"];
            if (!epoch_info.is_null() && epoch_info.contains("current_epoch_difficulty")) {
                uint32_t epoch_bits = epoch_info["current_epoch_difficulty"].get<uint32_t>();
                LOG_DEBUG("POOL", "Current epoch difficulty: ", epoch_bits, " bits");
            }
        }

        event_handler_->on_pool_status(status);
    }

    void PoolClient::handle_error(const Message &msg)
    {
        int error_code_int  = msg.payload.value("code", 0);
        ErrorCode code      = static_cast<ErrorCode>(error_code_int);
        std::string message = msg.payload.value("message", "Unknown error");
        LOG_ERROR("CLIENT", Color::RED, "Received error from server - code: ", error_code_int, ", message: ", message,
                  Color::RESET);
        // Handle specific error codes
        switch (code) {
            case ErrorCode::INVALID_MESSAGE:
            case ErrorCode::PROTOCOL_ERROR:
                // These are fatal errors, disconnect
                LOG_ERROR("CLIENT", "Fatal protocol error, disconnecting...");
                connected_     = false;
                authenticated_ = false;
                // Close the websocket
                if (ws_ && ws_->is_open()) {
                    beast::error_code ec;
                    ws_->close(websocket::close_code::protocol_error, ec);
                }
                break;
            case ErrorCode::AUTH_FAILED:
                authenticated_ = false;
                break;
            case ErrorCode::RATE_LIMITED:
                // Back off for a bit
                std::this_thread::sleep_for(std::chrono::seconds(5));
                break;
            default:
                break;
        }

        event_handler_->on_error(code, message);
    }

    void PoolClient::report_hashrate(const HashrateReportMessage &report)
    {
        if (!authenticated_.load()) {
            return;
        }

        Message msg;
        msg.type      = MessageType::HASHRATE_REPORT;
        msg.id        = Utils::generate_message_id();
        msg.timestamp = Utils::current_timestamp_ms();
        msg.payload   = report.to_json();

        send_message(msg);

        // Update local stats
        {
            std::lock_guard lock(stats_mutex_);
            worker_stats_.current_hashrate = report.hashrate;

            // Update average hashrate with exponential moving average
            const double alpha = 0.1;  // Smoothing factor
            if (worker_stats_.average_hashrate == 0) {
                worker_stats_.average_hashrate = report.hashrate;
            } else {
                worker_stats_.average_hashrate = alpha * report.hashrate + (1 - alpha) * worker_stats_.average_hashrate;
            }
        }
    }

    std::optional<PoolJob> PoolClient::get_current_job() const
    {
        std::lock_guard lock(jobs_mutex_);
        auto it = active_jobs_.find(current_job_id_);
        if (it != active_jobs_.end() && it->second.is_active && !it->second.is_expired()) {
            return it->second;
        }
        return std::nullopt;
    }

    std::vector<PoolJob> PoolClient::get_active_jobs() const
    {
        std::lock_guard lock(jobs_mutex_);
        std::vector<PoolJob> jobs;

        for (const auto &[id, job] : active_jobs_) {
            if (job.is_active && !job.is_expired()) {
                jobs.push_back(job);
            }
        }

        return jobs;
    }

    WorkerStats PoolClient::get_stats() const
    {
        std::lock_guard lock(stats_mutex_);
        return worker_stats_;
    }

    void PoolClient::cleanup_expired_jobs()
    {
        auto now = std::chrono::steady_clock::now();

        for (auto it = active_jobs_.begin(); it != active_jobs_.end();) {
            if (it->second.is_expired()) {
                event_handler_->on_job_cancelled(it->first);
                it = active_jobs_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void PoolClient::check_pending_timeouts()
    {
        std::lock_guard lock(pending_mutex_);
        auto now = std::chrono::steady_clock::now();

        for (auto it = pending_requests_.begin(); it != pending_requests_.end();) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second.timestamp).count();

            if (elapsed > config_.response_timeout_ms) {
                std::string msg_type_str;
                switch (it->second.type) {
                    case MessageType::AUTH:
                        msg_type_str = "AUTH";
                        break;
                    case MessageType::SUBMIT_SHARE:
                        msg_type_str = "SUBMIT_SHARE";
                        break;
                    case MessageType::GET_JOB:
                        msg_type_str = "GET_JOB";
                        break;
                    default:
                        msg_type_str = "UNKNOWN";
                }

                LOG_ERROR("CLIENT", "Request timeout: message_id=", it->first, ", type=", msg_type_str,
                          ", elapsed=", elapsed, "ms");
                it = pending_requests_.erase(it);
                // If we're getting timeouts, we might be disconnected
                if (!connected_.load() && running_.load()) {
                    bool should_reconnect =
                        config_.reconnect_attempts < 0 ||
                        reconnect_attempt_count_.load() < static_cast<unsigned>(config_.reconnect_attempts);

                    if (should_reconnect) {
                        LOG_INFO("CLIENT", "Connection seems lost (timeout), scheduling reconnect");
                        // Use the schedule_reconnect method if you implemented it, or:
                        std::thread([this]() {
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            reconnect();
                        }).detach();
                    }
                }
            } else {
                ++it;
            }
        }
    }

    bool PoolClient::is_reconnecting() const
    {
        return reconnecting_.load();
    }

    void PoolClient::reconnect()
    {
        bool expected = false;
        if (!reconnecting_.compare_exchange_strong(expected, true)) {
            LOG_DEBUG("CLIENT", "Already reconnecting, skipping");
            return;
        }

        if (!running_.load()) {
            reconnecting_ = false;
            return;
        }
        std::thread reconnect_thread([this]() {
            try {
                // Check max attempts
                LOG_DEBUG("CLIENT", "Checking reconnect attempts: current=", reconnect_attempt_count_.load(),
                          ", max=", config_.reconnect_attempts, " (",
                          config_.reconnect_attempts < 0 ? "infinite" : "limited", ")");
                if (config_.reconnect_attempts >= 0 &&
                    reconnect_attempt_count_ >= static_cast<unsigned>(config_.reconnect_attempts)) {
                    LOG_ERROR("CLIENT", "Maximum reconnection attempts (", config_.reconnect_attempts, ") exceeded");
                    running_      = false;
                    reconnecting_ = false;
                    return;
                }

                ++reconnect_attempt_count_;

                // Calculate delay with exponential backoff
                uint32_t delay = config_.reconnect_delay_ms;
                for (uint32_t i = 1; i < reconnect_attempt_count_; i++) {
                    delay = std::min(delay * 2, config_.max_reconnect_delay_ms);
                }

                LOG_INFO("CLIENT", "Reconnection attempt #", reconnect_attempt_count_.load(), " in ", delay, "ms...");

                std::this_thread::sleep_for(std::chrono::milliseconds(delay));

                // Check again if we should proceed
                if (!running_.load()) {
                    reconnecting_ = false;
                    return;
                }

                // Properly shutdown everything
                LOG_INFO("CLIENT", "Shutting down current connection...");

                // Signal threads to stop
                running_           = false;
                connected_         = false;
                authenticated_     = false;
                write_in_progress_ = false;

                // Clear session data
                session_id_.clear();

                // Stop work guard first
                if (work_guard_) {
                    work_guard_.reset();
                }

                // Shutdown connection cleanly
                shutdown_connection();

                // Wake up all condition variables
                incoming_cv_.notify_all();
                outgoing_cv_.notify_all();

                // Wait for threads to finish
                LOG_DEBUG("CLIENT", "Waiting for threads to finish...");
                // io_thread should exit quickly now
                if (io_thread_ && io_thread_->joinable()) {
                    io_thread_->join();
                    LOG_DEBUG("CLIENT", "io_thread joined successfully");
                }
                io_thread_.reset();

                // message_processor_thread
                if (message_processor_thread_.joinable()) {
                    incoming_cv_.notify_all();
                    message_processor_thread_.join();
                    LOG_DEBUG("CLIENT", "message_processor_thread joined successfully");
                }

                // keepalive_thread
                if (keepalive_thread_.joinable()) {
                    keepalive_thread_.join();
                    LOG_DEBUG("CLIENT", "keepalive_thread joined successfully");
                }

                LOG_DEBUG("CLIENT", "All threads stopped cleanly");

                // Clear websocket objects
                wss_.reset();
                ws_.reset();

                // Clear all queues
                {
                    std::lock_guard lock(outgoing_mutex_);
                    std::queue<Message> empty;
                    std::swap(outgoing_queue_, empty);
                }
                {
                    std::lock_guard lock(incoming_mutex_);
                    std::queue<Message> empty;
                    std::swap(incoming_queue_, empty);
                }
                {
                    std::lock_guard lock(pending_mutex_);
                    pending_requests_.clear();
                }

                // Reset io_context for fresh start
                ioc_.restart();
                // Reset work guard
                work_guard_ =
                    std::make_unique<net::executor_work_guard<net::io_context::executor_type>>(ioc_.get_executor());

                // Reset state for fresh connection
                running_ = true;  // Set running back to true for the new connection

                LOG_INFO("CLIENT", "Attempting to reconnect to ", config_.url, "...");

                // Call connect() to establish new connection
                if (connect()) {
                    LOG_INFO("CLIENT", Color::GREEN, "Reconnection successful!", Color::RESET);
                    reconnect_attempt_count_ = 0;
                    reconnecting_            = false;
                } else {
                    LOG_ERROR("CLIENT", "Reconnection failed");
                    reconnecting_ = false;
                    // The logic here needs adjustment - don't check running_ after connect() fails
                    bool should_retry = config_.reconnect_attempts < 0 ||
                                        reconnect_attempt_count_ < static_cast<unsigned>(config_.reconnect_attempts);
                    LOG_DEBUG("CLIENT", "Should retry? ", should_retry ? "YES" : "NO",
                              " (attempts=", reconnect_attempt_count_.load(), ", max=", config_.reconnect_attempts,
                              ")");

                    if (should_retry) {
                        // Remove "&& running_.load()" check here
                        LOG_INFO("CLIENT", "Scheduling another reconnect attempt...");
                        // Don't immediately retry - wait a bit
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        reconnect();
                    } else {
                        // Only NOW set running_ = false when we've exhausted all attempts
                        LOG_ERROR("CLIENT", "Giving up on reconnection");
                        running_ = false;
                        event_handler_->on_error(ErrorCode::CONNECTION_LOST, "Failed to reconnect after all attempts");
                    }
                }
            } catch (const std::exception &e) {
                LOG_ERROR("CLIENT", "Exception in reconnect thread: ", e.what());
                reconnecting_ = false;

                // Try again if appropriate
                bool should_retry =
                    running_.load() && (config_.reconnect_attempts < 0 ||
                                        reconnect_attempt_count_ < static_cast<unsigned>(config_.reconnect_attempts));

                if (should_retry) {
                    LOG_INFO("CLIENT", "Scheduling retry after exception");
                    // Schedule another attempt
                    std::thread([this]() {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
                        reconnect();
                    }).detach();
                } else {
                    running_ = false;
                    event_handler_->on_error(ErrorCode::CONNECTION_LOST,
                                             "Failed to reconnect after exception: " + std::string(e.what()));
                }
            }
        });

        reconnect_thread.detach();
    }

    // PoolClientManager implementation
    PoolClientManager::PoolClientManager() = default;

    PoolClientManager::~PoolClientManager()
    {
        disconnect_all();
    }

    bool PoolClientManager::add_pool(const std::string &name, const PoolConfig &config, IPoolEventHandler *handler)
    {
        std::lock_guard lock(mutex_);

        if (clients_.count(name) > 0) {
            LOG_ERROR("POOL_MGR", "Pool with name '", name, "' already exists");
            return false;
        }

        auto client    = std::make_shared<PoolClient>(config, handler);
        clients_[name] = client;

        if (primary_pool_name_.empty()) {
            primary_pool_name_ = name;
        }

        return true;
    }

    bool PoolClientManager::remove_pool(const std::string &name)
    {
        std::lock_guard lock(mutex_);

        auto it = clients_.find(name);
        if (it == clients_.end()) {
            return false;
        }

        it->second->disconnect();
        clients_.erase(it);

        if (primary_pool_name_ == name) {
            primary_pool_name_ = clients_.empty() ? "" : clients_.begin()->first;
        }

        return true;
    }

    bool PoolClientManager::set_primary_pool(const std::string &name)
    {
        std::lock_guard lock(mutex_);

        if (clients_.count(name) == 0) {
            return false;
        }

        primary_pool_name_ = name;
        return true;
    }

    std::string PoolClientManager::get_primary_pool() const
    {
        std::lock_guard lock(mutex_);
        return primary_pool_name_;
    }

    void PoolClientManager::enable_failover(bool enable)
    {
        std::lock_guard lock(mutex_);
        failover_enabled_ = enable;
    }

    void PoolClientManager::set_failover_order(const std::vector<std::string> &pool_names)
    {
        std::lock_guard lock(mutex_);
        failover_order_ = pool_names;
    }

    std::shared_ptr<PoolClient> PoolClientManager::get_client(const std::string &name) const
    {
        std::lock_guard lock(mutex_);

        auto it = clients_.find(name);
        if (it != clients_.end()) {
            return it->second;
        }

        return nullptr;
    }

    std::shared_ptr<PoolClient> PoolClientManager::get_primary_client() const
    {
        return get_client(get_primary_pool());
    }

    void PoolClientManager::connect_all()
    {
        std::lock_guard lock(mutex_);

        for (auto &[name, client] : clients_) {
            if (!client->is_connected()) {
                LOG_INFO("POOL_MGR", "Connecting to pool: ", name);
                client->connect();
            }
        }
    }

    void PoolClientManager::disconnect_all()
    {
        std::lock_guard lock(mutex_);

        for (auto &[name, client] : clients_) {
            if (client->is_connected()) {
                LOG_INFO("POOL_MGR", "Disconnecting from pool: ", name);
                client->disconnect();
            }
        }
    }

    std::map<std::string, WorkerStats> PoolClientManager::get_all_stats() const
    {
        std::lock_guard lock(mutex_);

        std::map<std::string, WorkerStats> all_stats;

        for (const auto &[name, client] : clients_) {
            if (client->is_connected()) {
                all_stats[name] = client->get_stats();
            }
        }

        return all_stats;
    }

    void PoolClientManager::handle_failover()
    {
        if (!failover_enabled_) {
            return;
        }

        // Check if primary pool is still connected
        if (const auto primary = get_primary_client(); primary && primary->is_connected()) {
            return;
        }

        // Try failover pools in order
        for (const auto &pool_name : failover_order_) {
            if (const auto client = get_client(pool_name); client && client->is_connected()) {
                LOG_INFO("POOL_MGR", "Failing over to pool: ", pool_name);
                set_primary_pool(pool_name);
                return;
            }
        }

        // If no pools in failover list are connected, try any connected pool
        std::lock_guard lock(mutex_);
        for (const auto &[name, client] : clients_) {
            if (client->is_connected() && name != primary_pool_name_) {
                LOG_INFO("POOL_MGR", "Failing over to pool: ", name);
                primary_pool_name_ = name;
                return;
            }
        }
    }
}  // namespace MiningPool

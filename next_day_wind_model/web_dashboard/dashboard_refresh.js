(function (root, factory) {
    "use strict";

    var api = factory();
    if (typeof module === "object" && module.exports) {
        module.exports = api;
    } else {
        root.WindDashboardRefresh = api;
    }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function metadataVersion(metadata) {
        if (!metadata) {
            return "";
        }
        return String(
            metadata.static_plot_generated_at_utc ||
            metadata.plot_refreshed_at_utc ||
            metadata.prediction_updated_at_utc ||
            metadata.prediction_generated_at_utc ||
            metadata.generated_at_utc ||
            ""
        );
    }

    function cacheBustedUrl(url, version, parameterName) {
        var key = parameterName || "v";
        var parsed = new URL(String(url), window.location.href);
        parsed.searchParams.set(key, String(version));
        return parsed.href;
    }

    function isNewerVersion(candidate, current) {
        if (!candidate || candidate === current) {
            return false;
        }
        if (!current) {
            return true;
        }
        var candidateTime = Date.parse(candidate);
        var currentTime = Date.parse(current);
        if (Number.isFinite(candidateTime) && Number.isFinite(currentTime)) {
            return candidateTime > currentTime;
        }
        return candidate !== current;
    }

    function shouldCheck(lastCheckAt, now, minimumIntervalMs, force) {
        return Boolean(force) || !lastCheckAt || now - lastCheckAt >= minimumIntervalMs;
    }

    function createController(options) {
        var metadataUrl = options.metadataUrl || "metadata_update.json";
        var minimumIntervalMs = Math.max(300000, Number(options.minimumIntervalMs) || 300000);
        var pollIntervalMs = Math.max(minimumIntervalMs, Number(options.pollIntervalMs) || minimumIntervalMs);
        var currentVersion = String(options.currentVersion || "");
        var refreshButton = options.refreshButton || null;
        var fetchImpl = options.fetchImpl || window.fetch.bind(window);
        var now = options.now || Date.now;
        var navigate = options.navigate || function (version) {
            window.location.replace(cacheBustedUrl(window.location.href, version, "dashboard_v"));
        };
        var lastCheckAt = now();
        var inFlight = null;
        var pollTimer = null;

        function setButtonState(isRefreshing) {
            if (!refreshButton) {
                return;
            }
            refreshButton.disabled = isRefreshing;
            refreshButton.textContent = isRefreshing ? "Refreshing\u2026" : "\u21bb Refresh";
            refreshButton.setAttribute("aria-busy", isRefreshing ? "true" : "false");
        }

        function checkForUpdate(settings) {
            var config = settings || {};
            var force = Boolean(config.force);
            var manual = Boolean(config.manual);
            var checkedAt = now();
            if (inFlight) {
                if (manual) {
                    setButtonState(true);
                    return inFlight.then(function (didNavigate) {
                        if (didNavigate) {
                            return true;
                        }
                        return checkForUpdate({ force: true, manual: true, reason: "manual-after-check" });
                    });
                }
                return inFlight;
            }
            if (!shouldCheck(lastCheckAt, checkedAt, minimumIntervalMs, force)) {
                return Promise.resolve(false);
            }
            lastCheckAt = checkedAt;
            setButtonState(manual);

            inFlight = fetchImpl(cacheBustedUrl(metadataUrl, checkedAt), {
                cache: "no-store",
                credentials: "same-origin",
                headers: { "Cache-Control": "no-cache" }
            })
                .then(function (response) {
                    if (!response.ok) {
                        throw new Error("metadata fetch failed: " + response.status);
                    }
                    return response.json();
                })
                .then(function (metadata) {
                    var nextVersion = metadataVersion(metadata);
                    if (manual || isNewerVersion(nextVersion, currentVersion)) {
                        navigate(nextVersion || checkedAt);
                        return true;
                    }
                    return false;
                })
                .catch(function (error) {
                    if (manual) {
                        navigate(checkedAt);
                        return true;
                    }
                    if (window.console && typeof window.console.debug === "function") {
                        window.console.debug("Dashboard freshness check skipped", error);
                    }
                    return false;
                })
                .finally(function () {
                    inFlight = null;
                    setButtonState(false);
                });
            return inFlight;
        }

        function handleVisibilityChange() {
            if (document.visibilityState === "visible") {
                checkForUpdate({ reason: "visibilitychange" });
            }
        }

        function handlePageShow() {
            checkForUpdate({ reason: "pageshow" });
        }

        function start() {
            if (refreshButton) {
                refreshButton.addEventListener("click", function () {
                    checkForUpdate({ force: true, manual: true, reason: "manual" });
                });
            }
            document.addEventListener("visibilitychange", handleVisibilityChange);
            window.addEventListener("pageshow", handlePageShow);
            pollTimer = window.setInterval(function () {
                if (document.visibilityState === "visible") {
                    checkForUpdate({ reason: "poll" });
                }
            }, pollIntervalMs);
            return api;
        }

        function stop() {
            document.removeEventListener("visibilitychange", handleVisibilityChange);
            window.removeEventListener("pageshow", handlePageShow);
            if (pollTimer !== null) {
                window.clearInterval(pollTimer);
                pollTimer = null;
            }
        }

        var api = {
            checkForUpdate: checkForUpdate,
            start: start,
            stop: stop
        };
        return api;
    }

    return {
        cacheBustedUrl: cacheBustedUrl,
        createController: createController,
        isNewerVersion: isNewerVersion,
        metadataVersion: metadataVersion,
        shouldCheck: shouldCheck
    };
});

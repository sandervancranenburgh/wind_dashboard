# Recommended nginx cache policy

The repository does not contain the live nginx configuration. Review the active
server layout before applying this example; do not copy it into `/etc/nginx`
without confirming that the document root and existing locations match.

```nginx
# Dashboard HTML must revalidate so foreground/manual refreshes receive the
# current asset-version references.
location = /index.html {
    expires -1;
    add_header Cache-Control "no-cache, must-revalidate" always;
    try_files $uri =404;
}

# Canonical freshness metadata and generated interactive data should never be
# satisfied from a stale browser/proxy cache.
location ~* ^/(metadata_update|current_day_interactive_data|next_day_interactive_data)\.json$ {
    expires -1;
    add_header Cache-Control "no-cache, no-store, must-revalidate" always;
    add_header Pragma "no-cache" always;
    try_files $uri =404;
}

# Generated forecast/performance plots keep stable filenames. Their HTML URLs
# carry ?v=<dashboard-generation>, while direct unversioned requests revalidate.
location ~* ^/(current_day_predictions(?:_mobile)?|next_day_predictions(?:_mobile)?|daily_mae_history(?:_mobile)?|model_gate_eval_history|model_gate_direction_spider|current_day_direction_spider)\.(?:png|svg)$ {
    expires -1;
    add_header Cache-Control "no-cache, must-revalidate" always;
    try_files $uri =404;
}
```

Leave the site's existing policy unchanged for other static files. In
particular, versioned CSS/JavaScript and the versioned icon assets can retain
their normal cache lifetime.

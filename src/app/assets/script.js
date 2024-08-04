window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        detect_scroll_inject: function() {
            var container = document.getElementById("scrollable_graph");
            container.addEventListener("scroll", function() {
                dash_clientside.set_props("scrollable_table_js_store", {children: container.scrollLeft})
            });
            return window.dash_clientside.no_update;
        },
        update_scroll: function(x) {
            return x
        }
    }
});
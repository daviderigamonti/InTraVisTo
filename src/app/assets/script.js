window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        detect_scroll_inject: function() {
            var table_container = document.getElementById("scrollable_table");
            var sankey_container = document.getElementById("scrollable_sankey");
            table_container.addEventListener("scroll", function() {
                dash_clientside.set_props("scrollable_table_js_store", {children: table_container.scrollLeft})
            });
            sankey_container.addEventListener("scroll", function() {
                dash_clientside.set_props("scrollable_sankey_js_store", {children: sankey_container.scrollLeft})
            });
            return window.dash_clientside.no_update;
        },
        update_table_scroll: function(x) {
            return x
        },
        update_sankey_scroll: function(x) {
            return x
        }
    }
});
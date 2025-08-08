
    {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "AQI Trend",
                "data": data,
                "borderColor": "#636EFA",
                "backgroundColor": "rgba(99, 110, 250, 0.2)",
                "fill": true
            }]
        },
        "options": {
            "scales": {
                "x": {
                    "title": {
                        "display": true,
                        "text": "Date"
                    }
                },
                "y": {
                    "title": {
                        "display": true,
                        "text": "AQI"
                    }
                }
            },
            "plugins": {
                "title": {
                    "display": true,
                    "text": "AQI Trend for " + city
                }
            }
        }
    }

"""Admin customisation for the demo app.

Everything declared shows up in the admin without being registered. Register
explicitly only to change how it is presented.
"""

from mlango import admin

from demo.datasets import Reviews


@admin.register(Reviews)
class ReviewsAdmin(admin.ObjectAdmin):
    list_display = ("id", "text", "label")
    list_filter = ("label",)
    search_fields = ("text",)
    list_per_page = 25

    def action_export(self, records):
        """Export the selected reviews as JSONL"""
        import json

        from mlango.storage import default_storage

        path = default_storage().path("exports/reviews.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        return f"Wrote {len(records)} review(s) to {path}"

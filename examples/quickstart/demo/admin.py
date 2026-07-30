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

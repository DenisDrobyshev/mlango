"""Admin customisation for the news app.

Declared objects appear in the admin automatically; register explicitly only to
change columns, filters or search.
"""

from mlango import admin  # noqa: F401

# @admin.register(MyDataset)
# class MyDatasetAdmin(admin.ObjectAdmin):
#     list_display = ("id", "text", "label")
#     list_filter = ("label",)
#     search_fields = ("text",)

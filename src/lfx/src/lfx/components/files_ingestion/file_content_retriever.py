"""Component that retrieves file content by path, for use as an agent tool.

Takes file data from a Read File component and allows an agent to look up
a specific file's content by providing its path.
"""

from __future__ import annotations

from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output, QueryInput
from lfx.schema.data import Data
from lfx.schema.message import Message


class FileContentRetrieverComponent(Component):
    display_name = "File Content Retriever"
    description = (
        "Retrieves the text content of a file given its path. "
        "Connect to an agent as a tool so it can read file contents."
    )
    icon = "file-text"
    name = "FileContentRetriever"
    add_tool_output = True

    inputs = [
        HandleInput(
            name="file_data",
            display_name="File Data",
            input_types=["Data", "DataFrame", "Message"],
            is_list=True,
            info="Output from a Read File component.",
        ),
        QueryInput(
            name="file_path",
            display_name="File Path",
            info="Path of the file to retrieve content for.",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            display_name="File Content",
            name="content",
            method="retrieve_content",
        ),
    ]

    def _build_file_map(self) -> dict[str, str]:
        """Build a lookup map from file paths to their text content."""
        from lfx.schema.dataframe import DataFrame

        file_map: dict[str, str] = {}
        for item in self.file_data:
            if isinstance(item, DataFrame):
                fp = item.attrs.get("source_file_path", "")
                if fp:
                    file_map[fp] = item.to_string()
            elif isinstance(item, Data):
                fp = item.data.get("file_path", "")
                text = item.get_text() or ""
                if fp:
                    file_map[fp] = text

        return file_map

    def retrieve_content(self) -> Message:
        file_map = self._build_file_map()
        query = self.file_path

        if not query:
            return Message(text="No file path provided.")

        content = file_map.get(query)

        if content is None:
            available = list(file_map.keys())
            return Message(text=f"File '{query}' not found. Available files: {available}")

        return Message(text=content)

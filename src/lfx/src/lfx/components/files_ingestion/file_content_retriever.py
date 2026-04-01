"""Component that retrieves file content by path, for use as an agent tool.

Takes file data from a Read File component and allows an agent to look up
a specific file's content by providing its path.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from lfx.custom.custom_component.component import Component
from lfx.io import HandleInput, Output, QueryInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cached_text_map: dict[str, str] | None = None
        self._cached_dataframe_map: dict[str, DataFrame] | None = None

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
            info=(
                "The full file path as a string (e.g., '/path/to/file.csv'). "
                "Do not pass search results or other objects."
            ),
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(
            display_name="File Content",
            name="content",
            method="retrieve_content",
            info="Retrieves file content as text. "
            "IMPORTANT: Pass ONLY the file path as a string argument (e.g., '/Users/name/document.txt'). "
            "Do NOT pass search results, Data objects, or other complex types. "
            "Returns: A Message containing the file's text content. "
            "Raises ValueError if file path is missing or file not found.",
            tool_mode=True,
        ),
        Output(
            display_name="Table",
            name="dataframe",
            method="retrieve_content_as_dataframe",
            info="Retrieves file content as a pandas DataFrame. "
            "IMPORTANT: Pass ONLY the file path as a string argument (e.g., '/Users/name/data.csv'). "
            "Do NOT pass search results, Data objects, or other complex types. "
            "Supported formats: CSV, Excel (.xlsx, .xls), Parquet, JSON, TSV. "
            "Returns: A DataFrame with the file's tabular data. "
            "Raises ValueError if file not found, unsupported format, or parsing fails.",
            tool_mode=True,
        ),
    ]

    def _get_file_maps(self) -> tuple[dict[str, str], dict[str, DataFrame]]:
        """Get cached file maps or build them if not cached.

        Returns:
            tuple: (text_map, dataframe_map)
                - text_map: file_path -> text content
                - dataframe_map: file_path -> DataFrame
        """
        # Return cached maps if available
        if self._cached_text_map is not None and self._cached_dataframe_map is not None:
            return self._cached_text_map, self._cached_dataframe_map

        # Build maps
        from lfx.schema.dataframe import DataFrame

        text_map: dict[str, str] = {}
        dataframe_map: dict[str, DataFrame] = {}

        for item in self.file_data:
            if isinstance(item, DataFrame):
                # Check attrs first
                fp = item.attrs.get("source_file_path", "")
                if fp:
                    text_map[fp] = item.to_string()
                    dataframe_map[fp] = item
                # Also check file_path column in DataFrame data
                elif not item.empty and "file_path" in item.columns:
                    # Get unique file paths from the DataFrame
                    unique_paths = item["file_path"].dropna().unique()
                    for path in unique_paths:
                        path_str = str(path)
                        if path_str:
                            text_map[path_str] = item.to_string()
                            dataframe_map[path_str] = item
            elif isinstance(item, Data):
                fp = item.data.get("file_path", "")
                text = item.get_text() or ""
                if fp:
                    text_map[fp] = text

        # Cache the maps
        self._cached_text_map = text_map
        self._cached_dataframe_map = dataframe_map

        return text_map, dataframe_map

    def retrieve_content(self) -> Message:
        """Retrieve file content as text.

        Returns:
            Message: The file content as text.

        Raises:
            ValueError: If file not found (only when called as a tool with a path).
        """
        text_map, _ = self._get_file_maps()
        query = self.file_path

        if not query:
            # During build phase, file_path may not be set yet
            # Return empty message to allow build to complete
            # When called as a tool, file_path will be provided by the agent
            return Message(text="")

        content = text_map.get(query)

        if content is None:
            available = list(text_map.keys())
            msg = f"File '{query}' not found. Available files: {available}"
            raise ValueError(msg)

        return Message(text=content)

    def retrieve_content_as_dataframe(self) -> DataFrame:
        """Retrieve file content as a DataFrame for tabular data files.

        Returns:
            DataFrame: The file content as a pandas DataFrame.

        Raises:
            ValueError: If no file path provided, file not found, or file type is not supported.
        """
        # Supported tabular file extensions
        tabular_extensions = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv"}

        query = self.file_path

        if not query:
            # During build phase, file_path may not be set yet
            # Return empty DataFrame to allow build to complete
            # When called as a tool, file_path will be provided by the agent
            return DataFrame(pd.DataFrame())

        # Check file extension
        file_ext = Path(query).suffix.lower()
        if file_ext not in tabular_extensions:
            supported = ", ".join(sorted(tabular_extensions))
            msg = (
                f"File type '{file_ext}' is not supported for DataFrame conversion. "
                f"Supported formats: {supported}. "
                f"File: '{query}'"
            )
            raise ValueError(msg)

        # Get cached maps (built once and reused)
        text_map, dataframe_map = self._get_file_maps()

        # Check if we have a DataFrame for this file
        if query in dataframe_map:
            return dataframe_map[query]

        # If not found as DataFrame, try to find as text and convert
        file_content = text_map.get(query)

        if file_content is None:
            available = list(text_map.keys())
            if available:
                msg = f"File '{query}' not found. Available files: {available}"
            else:
                msg = f"File '{query}' not found. No files available in the input data."
            raise ValueError(msg)

        # Convert file content to DataFrame based on file type
        try:
            if file_ext == ".csv":
                df = pd.read_csv(io.StringIO(file_content))
            elif file_ext == ".tsv":
                df = pd.read_csv(io.StringIO(file_content), sep="\t")
            elif file_ext in {".xlsx", ".xls"}:
                df = pd.read_excel(io.BytesIO(file_content.encode()))
            elif file_ext == ".parquet":
                df = pd.read_parquet(io.BytesIO(file_content.encode()))
            elif file_ext == ".json":
                df = pd.read_json(io.StringIO(file_content))
            else:
                msg = f"Unexpected file extension '{file_ext}' passed validation."
                raise ValueError(msg)
        except Exception as e:
            msg = f"Failed to parse file '{query}' as {file_ext.upper()} format. Error: {e!s}"
            raise ValueError(msg) from e

        # Convert to Langflow DataFrame and preserve file path in attrs
        result = DataFrame(df)
        result.attrs["source_file_path"] = query
        return result

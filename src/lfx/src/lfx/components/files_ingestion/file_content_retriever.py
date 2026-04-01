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
            tool_mode=True,
        ),
        Output(
            display_name="Table",
            name="dataframe",
            method="retrieve_content_as_dataframe",
            info="Retrieves file content as a DataFrame table. "
            "Input: File path (str) of a tabular data file (CSV, Excel, Parquet, JSON, or TSV). "
            "Example: 'data/sales.csv' or 'reports/metrics.xlsx'. "
            "Returns: A DataFrame containing the file's tabular data. "
            "Raises ValueError if the file is not a supported tabular format.",
            tool_mode=True,
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

    def retrieve_content_as_dataframe(self) -> DataFrame:
        """Retrieve file content as a DataFrame for tabular data files.

        Returns:
            DataFrame: The file content as a pandas DataFrame.

        Raises:
            ValueError: If no file path is provided, file not found, or file type is not supported.
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

        # First, check if we already have a DataFrame for this file
        for item in self.file_data:
            if isinstance(item, DataFrame):
                fp = item.attrs.get("source_file_path", "")
                if fp == query:
                    return item

        # If not found as DataFrame, try to find as Data and convert
        file_content = None
        for item in self.file_data:
            if isinstance(item, Data):
                fp = item.data.get("file_path", "")
                if fp == query:
                    file_content = item.get_text()
                    break

        if file_content is None:
            # Build list of available files for error message
            available_files = []
            for item in self.file_data:
                if isinstance(item, DataFrame):
                    fp = item.attrs.get("source_file_path", "")
                    if fp:
                        available_files.append(fp)
                elif isinstance(item, Data):
                    fp = item.data.get("file_path", "")
                    if fp:
                        available_files.append(fp)

            if available_files:
                msg = f"File '{query}' not found. Available files: {available_files}"
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
        else:
            # Convert to Langflow DataFrame and preserve file path in attrs
            result = DataFrame(df)
            result.attrs["source_file_path"] = query
            return result

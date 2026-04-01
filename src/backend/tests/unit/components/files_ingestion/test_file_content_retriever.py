"""Tests for FileContentRetrieverComponent.

Tests focus on real logic and edge cases:
- File lookup and retrieval from Data/DataFrame inputs
- DataFrame conversion for various file formats
- Error handling for missing files and invalid formats
- Edge cases with mixed inputs and malformed data
"""

import pandas as pd
import pytest
from lfx.components.files_ingestion.file_content_retriever import FileContentRetrieverComponent
from lfx.schema import DataFrame
from lfx.schema.data import Data
from lfx.schema.message import Message

from tests.base import ComponentTestBaseWithoutClient


class TestFileContentRetrieverComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        """Return the component class to test."""
        return FileContentRetrieverComponent

    @pytest.fixture
    def default_kwargs(self):
        """Return default kwargs with sample file data."""
        return {
            "file_data": [
                Data(text="content1", data={"file_path": "file1.txt"}),
                Data(text="content2", data={"file_path": "file2.txt"}),
            ],
            "file_path": "file1.txt",
        }

    @pytest.fixture
    def file_names_mapping(self):
        """Return empty list - new component without version history."""
        return []

    # ========== retrieve_content Tests ==========

    def test_retrieve_content_basic(self, component_class, default_kwargs):
        """Test basic file content retrieval."""
        component = component_class()
        component.set_attributes(default_kwargs)
        result = component.retrieve_content()

        assert isinstance(result, Message)
        assert result.text == "content1"

    def test_retrieve_content_second_file(self, component_class, default_kwargs):
        """Test retrieving different file from the map."""
        component = component_class()
        default_kwargs["file_path"] = "file2.txt"
        component.set_attributes(default_kwargs)
        result = component.retrieve_content()

        assert result.text == "content2"

    def test_retrieve_content_file_not_found(self, component_class, default_kwargs):
        """Test error message when file not found."""
        component = component_class()
        default_kwargs["file_path"] = "nonexistent.txt"
        component.set_attributes(default_kwargs)
        result = component.retrieve_content()

        assert isinstance(result, Message)
        assert "not found" in result.text
        assert "file1.txt" in result.text
        assert "file2.txt" in result.text

    def test_retrieve_content_empty_path(self, component_class, default_kwargs):
        """Test handling of empty file path."""
        component = component_class()
        default_kwargs["file_path"] = ""
        component.set_attributes(default_kwargs)
        result = component.retrieve_content()

        assert isinstance(result, Message)
        assert result.text == "No file path provided."

    def test_retrieve_content_whitespace_path(self, component_class, default_kwargs):
        """Test handling of whitespace-only file path."""
        component = component_class()
        default_kwargs["file_path"] = "   "
        component.set_attributes(default_kwargs)
        result = component.retrieve_content()

        assert isinstance(result, Message)
        # Whitespace path is treated as a file name, so it won't be found
        assert "not found" in result.text

    def test_retrieve_content_with_dataframe_input(self, component_class):
        """Test retrieving content from DataFrame input."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        langflow_df = DataFrame(df)
        langflow_df.attrs["source_file_path"] = "data.csv"

        component = component_class()
        component.set_attributes({"file_data": [langflow_df], "file_path": "data.csv"})
        result = component.retrieve_content()

        assert isinstance(result, Message)
        assert "col1" in result.text
        assert "col2" in result.text

    def test_retrieve_content_mixed_inputs(self, component_class):
        """Test with mixed Data and DataFrame inputs."""
        df = pd.DataFrame({"col1": [1, 2]})
        langflow_df = DataFrame(df)
        langflow_df.attrs["source_file_path"] = "data.csv"

        data_obj = Data(text="text content", data={"file_path": "file.txt"})

        component = component_class()
        component.set_attributes({"file_data": [langflow_df, data_obj], "file_path": "file.txt"})
        result = component.retrieve_content()

        assert result.text == "text content"

    def test_retrieve_content_data_without_file_path(self, component_class):
        """Test Data objects without file_path are ignored."""
        component = component_class()
        component.set_attributes(
            {
                "file_data": [
                    Data(text="no path", data={}),
                    Data(text="has path", data={"file_path": "valid.txt"}),
                ],
                "file_path": "valid.txt",
            }
        )
        result = component.retrieve_content()

        assert result.text == "has path"

    def test_retrieve_content_dataframe_without_source_path(self, component_class):
        """Test DataFrame without source_file_path is ignored."""
        df1 = DataFrame(pd.DataFrame({"col": [1]}))
        # No source_file_path set

        df2 = DataFrame(pd.DataFrame({"col": [2]}))
        df2.attrs["source_file_path"] = "valid.csv"

        component = component_class()
        component.set_attributes({"file_data": [df1, df2], "file_path": "valid.csv"})
        result = component.retrieve_content()

        assert "col" in result.text
        assert "2" in result.text

    # ========== as_dataframe Tests ==========

    def test_as_dataframe_csv_content(self, component_class):
        """Test converting CSV content to DataFrame."""
        csv_content = "col1,col2\n1,a\n2,b"
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=csv_content, data={"file_path": "data.csv"})], "file_path": "data.csv"}
        )

        result = component.as_dataframe()

        assert isinstance(result, DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]
        assert result["col1"].tolist() == [1, 2]
        assert result.attrs["source_file_path"] == "data.csv"

    def test_as_dataframe_tsv_content(self, component_class):
        """Test converting TSV content to DataFrame."""
        tsv_content = "col1\tcol2\n1\ta\n2\tb"
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=tsv_content, data={"file_path": "data.tsv"})], "file_path": "data.tsv"}
        )

        result = component.as_dataframe()

        assert isinstance(result, DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["col1", "col2"]

    def test_as_dataframe_json_content(self, component_class):
        """Test converting JSON content to DataFrame."""
        json_content = '[{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}]'
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=json_content, data={"file_path": "data.json"})], "file_path": "data.json"}
        )

        result = component.as_dataframe()

        assert isinstance(result, DataFrame)
        assert len(result) == 2
        assert "col1" in result.columns
        assert "col2" in result.columns

    def test_as_dataframe_returns_existing_dataframe(self, component_class):
        """Test that existing DataFrame is returned directly without conversion."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        langflow_df = DataFrame(df)
        langflow_df.attrs["source_file_path"] = "data.csv"

        component = component_class()
        component.set_attributes({"file_data": [langflow_df], "file_path": "data.csv"})

        result = component.as_dataframe()

        assert result is langflow_df
        assert len(result) == 3

    def test_as_dataframe_no_file_path_raises(self, component_class):
        """Test that missing file path raises ValueError."""
        component = component_class()
        component.set_attributes({"file_data": [Data(text="content", data={"file_path": "file.csv"})], "file_path": ""})

        with pytest.raises(ValueError, match="No file path provided"):
            component.as_dataframe()

    def test_as_dataframe_unsupported_extension_raises(self, component_class):
        """Test that unsupported file extension raises ValueError."""
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text="content", data={"file_path": "file.txt"})], "file_path": "file.txt"}
        )

        with pytest.raises(ValueError, match="not supported for DataFrame conversion"):
            component.as_dataframe()

    def test_as_dataframe_file_not_found_raises(self, component_class):
        """Test that missing file raises ValueError with available files."""
        component = component_class()
        component.set_attributes(
            {
                "file_data": [
                    Data(text="content1", data={"file_path": "file1.csv"}),
                    Data(text="content2", data={"file_path": "file2.csv"}),
                ],
                "file_path": "missing.csv",
            }
        )

        with pytest.raises(ValueError, match=r"not found.*Available files"):
            component.as_dataframe()

    def test_as_dataframe_no_files_available_raises(self, component_class):
        """Test error message when no files available."""
        component = component_class()
        component.set_attributes({"file_data": [], "file_path": "missing.csv"})

        with pytest.raises(ValueError, match="No files available in the input data"):
            component.as_dataframe()

    def test_as_dataframe_malformed_json_raises(self, component_class):
        """Test that malformed JSON raises ValueError."""
        malformed_json = '{"col1": 1, "col2": "unclosed'
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=malformed_json, data={"file_path": "bad.json"})], "file_path": "bad.json"}
        )

        with pytest.raises(ValueError, match=r"Failed to parse file.*JSON"):
            component.as_dataframe()

    def test_as_dataframe_supported_extensions(self, component_class):
        """Test all supported extensions are recognized."""
        supported = [".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv"]

        for ext in supported:
            csv_content = "col1,col2\n1,a"
            component = component_class()
            component.set_attributes(
                {"file_data": [Data(text=csv_content, data={"file_path": f"file{ext}"})], "file_path": f"file{ext}"}
            )

            # Should not raise for supported extensions
            # (actual parsing may fail for non-CSV content, but extension check passes)
            try:
                component.as_dataframe()
            except ValueError as e:
                # If it fails, it should be parsing error, not extension error
                if "not supported for DataFrame conversion" in str(e):
                    pytest.fail(f"Extension check failed for {ext}: {e}")

    def test_as_dataframe_case_insensitive_extension(self, component_class):
        """Test that file extensions are case-insensitive."""
        csv_content = "col1,col2\n1,a"
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=csv_content, data={"file_path": "file.CSV"})], "file_path": "file.CSV"}
        )

        result = component.as_dataframe()
        assert isinstance(result, DataFrame)

    def test_as_dataframe_empty_csv(self, component_class):
        """Test handling of empty CSV file."""
        empty_csv = "col1,col2\n"
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=empty_csv, data={"file_path": "empty.csv"})], "file_path": "empty.csv"}
        )

        result = component.as_dataframe()
        assert isinstance(result, DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2"]

    def test_as_dataframe_csv_with_special_characters(self, component_class):
        """Test CSV with special characters and quotes."""
        csv_content = 'name,description\n"John, Doe","He said ""hello"""'
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=csv_content, data={"file_path": "special.csv"})], "file_path": "special.csv"}
        )

        result = component.as_dataframe()
        assert isinstance(result, DataFrame)
        assert len(result) == 1
        assert result["name"].iloc[0] == "John, Doe"

    def test_as_dataframe_json_nested_structure(self, component_class):
        """Test JSON with nested structures."""
        json_content = '[{"id": 1, "data": {"nested": "value"}}, {"id": 2, "data": {"nested": "value2"}}]'
        component = component_class()
        component.set_attributes(
            {"file_data": [Data(text=json_content, data={"file_path": "nested.json"})], "file_path": "nested.json"}
        )

        result = component.as_dataframe()
        assert isinstance(result, DataFrame)
        assert len(result) == 2

    # ========== _build_file_map Tests ==========

    def test_build_file_map_empty_input(self, component_class):
        """Test building file map with empty input."""
        component = component_class()
        component.set_attributes({"file_data": [], "file_path": ""})

        file_map = component._build_file_map()
        assert file_map == {}

    def test_build_file_map_data_only(self, component_class):
        """Test building file map with only Data objects."""
        component = component_class()
        component.set_attributes(
            {
                "file_data": [
                    Data(text="content1", data={"file_path": "file1.txt"}),
                    Data(text="content2", data={"file_path": "file2.txt"}),
                ],
                "file_path": "",
            }
        )

        file_map = component._build_file_map()
        assert len(file_map) == 2
        assert file_map["file1.txt"] == "content1"
        assert file_map["file2.txt"] == "content2"

    def test_build_file_map_dataframe_only(self, component_class):
        """Test building file map with only DataFrame objects."""
        df1 = DataFrame(pd.DataFrame({"col": [1]}))
        df1.attrs["source_file_path"] = "data1.csv"

        df2 = DataFrame(pd.DataFrame({"col": [2]}))
        df2.attrs["source_file_path"] = "data2.csv"

        component = component_class()
        component.set_attributes({"file_data": [df1, df2], "file_path": ""})

        file_map = component._build_file_map()
        assert len(file_map) == 2
        assert "data1.csv" in file_map
        assert "data2.csv" in file_map

    def test_build_file_map_data_with_empty_text(self, component_class):
        """Test Data objects with empty text are included."""
        component = component_class()
        component.set_attributes({"file_data": [Data(text="", data={"file_path": "empty.txt"})], "file_path": ""})

        file_map = component._build_file_map()
        assert "empty.txt" in file_map
        assert file_map["empty.txt"] == ""

    def test_build_file_map_data_with_none_text(self, component_class):
        """Test Data objects with None text are handled."""
        component = component_class()
        component.set_attributes({"file_data": [Data(text=None, data={"file_path": "none.txt"})], "file_path": ""})

        file_map = component._build_file_map()
        assert "none.txt" in file_map
        assert file_map["none.txt"] == ""

    def test_build_file_map_duplicate_paths_last_wins(self, component_class):
        """Test that duplicate file paths use the last occurrence."""
        component = component_class()
        component.set_attributes(
            {
                "file_data": [
                    Data(text="first", data={"file_path": "dup.txt"}),
                    Data(text="second", data={"file_path": "dup.txt"}),
                ],
                "file_path": "",
            }
        )

        file_map = component._build_file_map()
        assert file_map["dup.txt"] == "second"

    def test_build_file_map_mixed_with_and_without_paths(self, component_class):
        """Test mixed inputs where some have paths and some don't."""
        df_with_path = DataFrame(pd.DataFrame({"col": [1]}))
        df_with_path.attrs["source_file_path"] = "has_path.csv"

        df_without_path = DataFrame(pd.DataFrame({"col": [2]}))
        # No source_file_path

        component = component_class()
        component.set_attributes(
            {
                "file_data": [
                    Data(text="has path", data={"file_path": "file.txt"}),
                    Data(text="no path", data={}),
                    df_with_path,
                    df_without_path,
                ],
                "file_path": "",
            }
        )

        file_map = component._build_file_map()
        assert len(file_map) == 2
        assert "file.txt" in file_map
        assert "has_path.csv" in file_map


# Made with Bob

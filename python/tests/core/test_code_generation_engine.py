#
#   Copyright 2024 Hopsworks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""Tests for the code generation engine module."""

from __future__ import annotations

import pytest
from hsfs.core.code_generation_engine import (
    CodeGenerationEngine,
    GeneratedCode,
    UDFCodeGenerator,
    UDFContext,
    get_code_generation_engine,
    get_udf_code_generator,
)


class TestGeneratedCode:
    """Tests for the GeneratedCode dataclass."""

    def test_generated_code_creation(self):
        """Test basic GeneratedCode creation."""
        code = "def foo(): return 42"
        generated = GeneratedCode(code=code, template_name="test_template")

        assert generated.code == code
        assert generated.template_name == "test_template"
        assert len(generated.context_hash) == 8  # MD5 hash truncated to 8 chars

    def test_generated_code_custom_hash(self):
        """Test GeneratedCode with custom hash."""
        generated = GeneratedCode(
            code="def foo(): pass",
            template_name="test",
            context_hash="custom123",
        )

        assert generated.context_hash == "custom123"

    def test_generated_code_auto_hash_deterministic(self):
        """Test that auto-generated hash is deterministic."""
        code = "def bar(): return 'hello'"
        gen1 = GeneratedCode(code=code, template_name="test")
        gen2 = GeneratedCode(code=code, template_name="test")

        assert gen1.context_hash == gen2.context_hash


class TestUDFContext:
    """Tests for the UDFContext dataclass."""

    def test_udf_context_minimal(self):
        """Test UDFContext with minimal parameters."""
        context = UDFContext(
            function_name="my_udf",
            formatted_function_source="def my_udf(x):\n    return x + 1",
            module_imports="",
            output_column_names=["result"],
            return_types=["double"],
        )

        assert context.function_name == "my_udf"
        assert context.output_column_names == ["result"]
        assert context.return_types == ["double"]
        assert context.datetime_output_indices == []
        assert context.datetime_output_columns == []

    def test_udf_context_with_datetime(self):
        """Test UDFContext with datetime outputs."""
        context = UDFContext(
            function_name="my_udf",
            formatted_function_source="def my_udf(x):\n    return x",
            module_imports="",
            output_column_names=["ts_col"],
            return_types=["timestamp"],
            datetime_output_indices=[0],
            datetime_output_columns=["ts_col"],
        )

        assert context.datetime_output_indices == [0]
        assert context.datetime_output_columns == ["ts_col"]

    def test_udf_context_to_template_context_single_output(self):
        """Test to_template_context for single output."""
        context = UDFContext(
            function_name="add_one",
            formatted_function_source="def add_one(x):\n    return x + 1",
            module_imports="import numpy as np",
            output_column_names=["result"],
            return_types=["double"],
        )

        template_ctx = context.to_template_context()

        assert template_ctx["function_name"] == "add_one"
        assert template_ctx["module_imports"] == "import numpy as np"
        assert template_ctx["has_datetime_outputs"] is False
        assert template_ctx["has_multiple_outputs"] is False

    def test_udf_context_to_template_context_multiple_outputs(self):
        """Test to_template_context for multiple outputs."""
        context = UDFContext(
            function_name="multi_out",
            formatted_function_source="def multi_out(x):\n    return x, x*2",
            module_imports="",
            output_column_names=["a", "b"],
            return_types=["double", "double"],
        )

        template_ctx = context.to_template_context()

        assert template_ctx["has_multiple_outputs"] is True

    def test_udf_context_to_template_context_with_extra(self):
        """Test to_template_context with extra context variables."""
        context = UDFContext(
            function_name="my_func",
            formatted_function_source="def my_func(x):\n    return x",
            module_imports="",
            output_column_names=["out"],
            return_types=["string"],
        )

        template_ctx = context.to_template_context(
            rename_outputs=True,
            custom_var="custom_value",
        )

        assert template_ctx["rename_outputs"] is True
        assert template_ctx["custom_var"] == "custom_value"


class TestCodeGenerationEngine:
    """Tests for the CodeGenerationEngine class."""

    def test_engine_creation(self):
        """Test engine can be created."""
        engine = CodeGenerationEngine()
        assert engine is not None

    def test_render_string_simple(self):
        """Test rendering a simple template string."""
        engine = CodeGenerationEngine()
        result = engine.render_string(
            "def {{ name }}(): return {{ value }}",
            {"name": "foo", "value": 42},
        )

        assert result == "def foo(): return 42"

    def test_render_string_with_conditional(self):
        """Test rendering with Jinja2 conditionals."""
        engine = CodeGenerationEngine()
        template = "{% if show_return %}return {{ value }}{% endif %}"

        result_true = engine.render_string(template, {"show_return": True, "value": 1})
        result_false = engine.render_string(
            template, {"show_return": False, "value": 1}
        )

        assert result_true == "return 1"
        assert result_false == ""

    def test_render_string_with_loop(self):
        """Test rendering with Jinja2 loops."""
        engine = CodeGenerationEngine()
        template = (
            "{% for i in items %}{{ i }}{% if not loop.last %}, {% endif %}{% endfor %}"
        )

        result = engine.render_string(template, {"items": [1, 2, 3]})

        assert result == "1, 2, 3"

    def test_render_template_python_udf_wrapper(self):
        """Test rendering the registered python_udf_wrapper template."""
        engine = CodeGenerationEngine()
        context = {
            "module_imports": "",
            "has_datetime_outputs": False,
            "timezone_converter": "",
            "formatted_function_source": "def my_udf(x):\n    return x + 1",
            "function_name": "my_udf",
            "has_multiple_outputs": False,
            "rename_outputs": False,
            "datetime_output_indices": [],
        }

        result = engine.render_template("python_udf_wrapper", context)

        assert "def wrapper(*args):" in result
        assert "_transformed_features = my_udf(*args)" in result
        assert "return _transformed_features" in result

    def test_render_template_pandas_udf_wrapper(self):
        """Test rendering the registered pandas_udf_wrapper template."""
        engine = CodeGenerationEngine()
        context = {
            "module_imports": "",
            "timezone_converter": "def _convert_timezone(x): pass",
            "formatted_function_source": "def my_udf(x):\n    return x",
            "function_name": "my_udf",
            "has_multiple_outputs": False,
            "datetime_output_columns": [],
        }

        result = engine.render_template("pandas_udf_wrapper", context)

        assert "def renaming_wrapper(*args):" in result
        assert "import pandas as pd" in result

    def test_render_template_not_found(self):
        """Test that unknown template raises TemplateNotFound."""
        from jinja2 import TemplateNotFound

        engine = CodeGenerationEngine()

        with pytest.raises(TemplateNotFound):
            engine.render_template("nonexistent_template", {})

    def test_get_template_content(self):
        """Test getting template content from file."""
        engine = CodeGenerationEngine()
        template = engine.get_template_content("python_udf_wrapper")

        assert "def wrapper(*args):" in template

    def test_get_template_content_not_found(self):
        """Test that unknown template raises FileNotFoundError."""
        engine = CodeGenerationEngine()

        with pytest.raises(FileNotFoundError):
            engine.get_template_content("nonexistent")

    def test_list_templates(self):
        """Test listing available templates."""
        engine = CodeGenerationEngine()
        templates = engine.list_templates()

        assert "python_udf_wrapper" in templates
        assert "pandas_udf_wrapper" in templates
        assert "python_timezone_converter" in templates
        assert "pandas_timezone_converter" in templates

    def test_templates_dir_property(self):
        """Test templates_dir property returns correct path."""
        engine = CodeGenerationEngine()

        assert engine.templates_dir.exists()
        assert engine.templates_dir.name == "templates"


class TestUDFCodeGenerator:
    """Tests for the UDFCodeGenerator class."""

    def test_generator_creation(self):
        """Test generator can be created."""
        generator = UDFCodeGenerator()
        assert generator is not None
        assert generator.engine is not None

    def test_generator_with_custom_engine(self):
        """Test generator with custom engine."""
        engine = CodeGenerationEngine()
        generator = UDFCodeGenerator(engine)

        assert generator.engine is engine

    def test_generate_python_wrapper_single_output(self):
        """Test generating Python wrapper for single output."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="add_one",
            formatted_function_source="def add_one(x):\n    return x + 1",
            module_imports="",
            output_column_names=["result"],
            return_types=["double"],
        )

        generated = generator.generate_python_wrapper(context, rename_outputs=False)

        assert isinstance(generated, GeneratedCode)
        assert generated.template_name == "python_udf_wrapper"
        assert "def wrapper(*args):" in generated.code
        assert "return _transformed_features" in generated.code

    def test_generate_python_wrapper_multiple_outputs_rename(self):
        """Test generating Python wrapper with multiple outputs and renaming."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="multi",
            formatted_function_source="def multi(x):\n    return x, x*2",
            module_imports="",
            output_column_names=["a", "b"],
            return_types=["double", "double"],
        )

        generated = generator.generate_python_wrapper(context, rename_outputs=True)

        assert (
            "return dict(zip(_output_col_names, _transformed_features))"
            in generated.code
        )

    def test_generate_python_wrapper_with_datetime(self):
        """Test generating Python wrapper with datetime outputs."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="ts_func",
            formatted_function_source="def ts_func(x):\n    return x",
            module_imports="",
            output_column_names=["ts_col"],
            return_types=["timestamp"],
            datetime_output_indices=[0],
        )

        generated = generator.generate_python_wrapper(context, rename_outputs=False)

        assert "_convert_timezone" in generated.code
        assert "tzlocal" in generated.code

    def test_generate_pandas_wrapper_single_output(self):
        """Test generating Pandas wrapper for single output."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="normalize",
            formatted_function_source="def normalize(x):\n    return x / x.max()",
            module_imports="",
            output_column_names=["normalized"],
            return_types=["double"],
        )

        generated = generator.generate_pandas_wrapper(context)

        assert isinstance(generated, GeneratedCode)
        assert generated.template_name == "pandas_udf_wrapper"
        assert "def renaming_wrapper(*args):" in generated.code
        assert "import pandas as pd" in generated.code

    def test_generate_pandas_wrapper_multiple_outputs(self):
        """Test generating Pandas wrapper for multiple outputs."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="split",
            formatted_function_source="def split(x):\n    return x, x*2",
            module_imports="",
            output_column_names=["col_a", "col_b"],
            return_types=["double", "double"],
        )

        generated = generator.generate_pandas_wrapper(context)

        assert "_df.columns = _output_col_names" in generated.code
        assert "pd.concat" in generated.code

    def test_generate_pandas_wrapper_with_datetime(self):
        """Test generating Pandas wrapper with datetime columns."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="ts_transform",
            formatted_function_source="def ts_transform(x):\n    return x",
            module_imports="",
            output_column_names=["timestamp_col"],
            return_types=["timestamp"],
            datetime_output_columns=["timestamp_col"],
        )

        generated = generator.generate_pandas_wrapper(context)

        assert "_convert_timezone" in generated.code
        assert "tz_localize" in generated.code

    def test_generate_default_method(self):
        """Test the default generate method."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="test_func",
            formatted_function_source="def test_func(x):\n    return x",
            module_imports="",
            output_column_names=["out"],
            return_types=["string"],
        )

        generated = generator.generate(context)

        # Default should use Python wrapper without renaming
        assert "def wrapper(*args):" in generated.code

    def test_generate_aggregation_wrapper_single_output(self):
        """Test generating aggregation wrapper for single output."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="calc_mean",
            formatted_function_source="def calc_mean(values):\n    return values.mean()",
            module_imports="",
            output_column_names=["mean_value"],
            return_types=["double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        assert isinstance(generated, GeneratedCode)
        assert generated.template_name == "aggregation_udf_wrapper"
        assert "def aggregation_wrapper(*args):" in generated.code
        assert "import pandas as pd" in generated.code

    def test_generate_aggregation_wrapper_multiple_outputs(self):
        """Test generating aggregation wrapper for multiple outputs."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="calc_stats",
            formatted_function_source="def calc_stats(values):\n    return values.mean(), values.std()",
            module_imports="",
            output_column_names=["mean", "std"],
            return_types=["double", "double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        assert "isinstance(_result, tuple)" in generated.code
        assert "isinstance(_result, list)" in generated.code

    def test_generate_aggregation_wrapper_with_datetime(self):
        """Test generating aggregation wrapper with datetime columns."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="get_latest",
            formatted_function_source="def get_latest(timestamps):\n    return timestamps.max()",
            module_imports="",
            output_column_names=["latest_ts"],
            return_types=["timestamp"],
            datetime_output_columns=["latest_ts"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        assert "_convert_timezone" in generated.code
        assert "tz_localize" in generated.code


class TestSingletonFunctions:
    """Tests for singleton/factory functions."""

    def test_get_code_generation_engine(self):
        """Test getting the default engine singleton."""
        engine1 = get_code_generation_engine()
        engine2 = get_code_generation_engine()

        assert engine1 is engine2  # Same instance

    def test_get_udf_code_generator(self):
        """Test getting a UDF code generator."""
        generator = get_udf_code_generator()

        assert isinstance(generator, UDFCodeGenerator)


class TestGeneratedCodeExecution:
    """Tests that verify generated code can be executed."""

    def test_python_wrapper_executes_correctly(self):
        """Test that generated Python wrapper code actually executes."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="double_value",
            formatted_function_source="def double_value(x):\n    return x * 2",
            module_imports="",
            output_column_names=["doubled"],
            return_types=["double"],
        )

        generated = generator.generate_python_wrapper(context, rename_outputs=False)

        # Execute the generated code
        scope = {"_output_col_names": ["doubled"]}
        exec(generated.code, scope)
        wrapper = scope["wrapper"]

        # Test the wrapper
        result = wrapper(5)
        assert result == 10

    def test_python_wrapper_multiple_outputs_executes(self):
        """Test that generated Python wrapper with multiple outputs executes."""
        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="split_triple",
            formatted_function_source="def split_triple(x):\n    return x, x*2, x*3",
            module_imports="",
            output_column_names=["a", "b", "c"],
            return_types=["double", "double", "double"],
        )

        generated = generator.generate_python_wrapper(context, rename_outputs=True)

        # Execute the generated code
        scope = {"_output_col_names": ["a", "b", "c"]}
        exec(generated.code, scope)
        wrapper = scope["wrapper"]

        result = wrapper(5)
        assert result == {"a": 5, "b": 10, "c": 15}

    def test_pandas_wrapper_executes_correctly(self):
        """Test that generated Pandas wrapper code actually executes."""
        import pandas as pd

        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="add_ten",
            formatted_function_source="def add_ten(x):\n    return x + 10",
            module_imports="",
            output_column_names=["plus_ten"],
            return_types=["double"],
        )

        generated = generator.generate_pandas_wrapper(context)

        # Execute the generated code
        scope = {
            "_output_col_names": ["plus_ten"],
            "_datetime_output_columns": [],
        }
        exec(generated.code, scope)
        wrapper = scope["renaming_wrapper"]

        # Test with pandas Series
        input_series = pd.Series([1, 2, 3])
        result = wrapper(input_series)

        assert isinstance(result, pd.Series)
        assert result.name == "plus_ten"
        assert list(result) == [11, 12, 13]

    def test_aggregation_wrapper_scalar_output_executes(self):
        """Test aggregation wrapper with scalar output."""
        import pandas as pd

        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="calc_mean",
            formatted_function_source="def calc_mean(values):\n    return values.mean()",
            module_imports="",
            output_column_names=["mean_value"],
            return_types=["double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        # Execute the generated code
        scope = {
            "_output_col_names": ["mean_value"],
            "_datetime_output_columns": [],
        }
        exec(generated.code, scope)
        wrapper = scope["aggregation_wrapper"]

        # Test with pandas Series
        input_series = pd.Series([10.0, 20.0, 30.0])
        result = wrapper(input_series)

        assert isinstance(result, pd.DataFrame)
        assert "mean_value" in result.columns
        assert result["mean_value"].iloc[0] == 20.0

    def test_aggregation_wrapper_tuple_output_executes(self):
        """Test aggregation wrapper with tuple output."""
        import pandas as pd

        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="calc_stats",
            formatted_function_source="def calc_stats(values):\n    return values.mean(), values.sum()",
            module_imports="",
            output_column_names=["mean", "total"],
            return_types=["double", "double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        # Execute the generated code
        scope = {
            "_output_col_names": ["mean", "total"],
            "_datetime_output_columns": [],
        }
        exec(generated.code, scope)
        wrapper = scope["aggregation_wrapper"]

        # Test with pandas Series
        input_series = pd.Series([10.0, 20.0, 30.0])
        result = wrapper(input_series)

        assert isinstance(result, pd.DataFrame)
        assert "mean" in result.columns
        assert "total" in result.columns
        assert result["mean"].iloc[0] == 20.0
        assert result["total"].iloc[0] == 60.0

    def test_aggregation_wrapper_dataframe_output_executes(self):
        """Test aggregation wrapper with DataFrame output."""
        import pandas as pd

        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="calc_stats_df",
            formatted_function_source="def calc_stats_df(values):\n    return pd.DataFrame({'col_0': [values.mean()], 'col_1': [values.std()]})",
            module_imports="import pandas as pd",
            output_column_names=["avg", "stddev"],
            return_types=["double", "double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        # Execute the generated code
        scope = {
            "_output_col_names": ["avg", "stddev"],
            "_datetime_output_columns": [],
        }
        exec(generated.code, scope)
        wrapper = scope["aggregation_wrapper"]

        # Test with pandas Series
        input_series = pd.Series([10.0, 20.0, 30.0])
        result = wrapper(input_series)

        assert isinstance(result, pd.DataFrame)
        assert "avg" in result.columns
        assert "stddev" in result.columns
        assert result["avg"].iloc[0] == 20.0

    def test_aggregation_wrapper_series_output_executes(self):
        """Test aggregation wrapper with Series output."""
        import pandas as pd

        generator = UDFCodeGenerator()
        context = UDFContext(
            function_name="get_first",
            formatted_function_source="def get_first(values):\n    return pd.Series([values.iloc[0]])",
            module_imports="import pandas as pd",
            output_column_names=["first_value"],
            return_types=["double"],
        )

        generated = generator.generate_aggregation_wrapper(context)

        # Execute the generated code
        scope = {
            "_output_col_names": ["first_value"],
            "_datetime_output_columns": [],
        }
        exec(generated.code, scope)
        wrapper = scope["aggregation_wrapper"]

        # Test with pandas Series
        input_series = pd.Series([10.0, 20.0, 30.0])
        result = wrapper(input_series)

        assert isinstance(result, pd.DataFrame)
        assert "first_value" in result.columns
        assert result["first_value"].iloc[0] == 10.0

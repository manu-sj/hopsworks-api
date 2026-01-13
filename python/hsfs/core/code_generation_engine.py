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
r"""Code generation engine for dynamically generating Python code using Jinja2 templates.

This module provides a centralized code generation engine that uses Jinja2 templates
to generate Python code dynamically. It is primarily used for generating UDF wrapper
functions but can be extended for other code generation needs.

Templates are stored in the `templates/` directory relative to this module.

Example:
    ```python
    from hsfs.core.code_generation_engine import CodeGenerationEngine, UDFCodeGenerator

    engine = CodeGenerationEngine()
    generator = UDFCodeGenerator(engine)

    context = UDFContext(
        function_name="my_udf",
        formatted_function_source="def my_udf(x):\n    return x + 1",
        module_imports="",
        output_column_names=["result"],
        return_types=["double"],
    )
    generated = generator.generate_python_wrapper(context, rename_outputs=False)
    print(generated.code)
    ```
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


# Path to templates directory (relative to this module)
TEMPLATES_DIR = Path(__file__).parent / "templates"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class GeneratedCode:
    """Result of code generation.

    Contains the generated code string along with metadata about the generation
    process for debugging and caching purposes.

    Parameters:
        code: The generated Python code as a string.
        template_name: Name of the template used to generate the code.
        context_hash: Hash of the context used for generation (useful for caching).
    """

    code: str
    template_name: str
    context_hash: str = ""

    def __post_init__(self) -> None:
        if not self.context_hash:
            self.context_hash = hashlib.md5(
                self.code.encode(), usedforsecurity=False
            ).hexdigest()[:8]


@dataclass
class UDFContext:
    """Context for UDF code generation.

    Contains all the information needed to generate a UDF wrapper function.

    Parameters:
        function_name: Name of the original UDF function.
        formatted_function_source: The formatted source code of the UDF (without decorators/type hints).
        module_imports: Import statements required by the UDF.
        output_column_names: Names of the output columns from the UDF.
        return_types: List of return type strings (e.g., ['string', 'double']).
        datetime_output_indices: Indices of outputs that are datetime types (for Python wrapper).
        datetime_output_columns: Column names that are datetime types (for Pandas wrapper).
    """

    function_name: str
    formatted_function_source: str
    module_imports: str
    output_column_names: list[str]
    return_types: list[str]
    datetime_output_indices: list[int] = field(default_factory=list)
    datetime_output_columns: list[str] = field(default_factory=list)

    def to_template_context(self, **extra_context: Any) -> dict[str, Any]:
        """Convert context to dictionary for template rendering.

        Parameters:
            **extra_context: Additional context variables to include.

        Returns:
            Dictionary suitable for Jinja2 template rendering.
        """
        context = {
            "function_name": self.function_name,
            "formatted_function_source": self.formatted_function_source,
            "module_imports": self.module_imports,
            "output_column_names": self.output_column_names,
            "return_types": self.return_types,
            "datetime_output_indices": self.datetime_output_indices,
            "datetime_output_columns": self.datetime_output_columns,
            "has_datetime_outputs": len(self.datetime_output_indices) > 0
            or len(self.datetime_output_columns) > 0,
            "has_multiple_outputs": len(self.return_types) > 1,
        }
        context.update(extra_context)
        return context


# =============================================================================
# Code Generation Engine
# =============================================================================


class CodeGenerationEngine:
    """Central engine for code generation using Jinja2 templates.

    This class manages a Jinja2 environment configured for Python code generation.
    Templates are loaded from the `templates/` directory by default.

    The engine is designed to be used as a singleton or instantiated as needed
    by code generators.

    Example:
        ```python
        engine = CodeGenerationEngine()
        code = engine.render_template(
            "python_udf_wrapper",
            {"function_name": "foo", "formatted_function_source": "..."}
        )
        ```
    """

    def __init__(self, templates_dir: Path | str | None = None) -> None:
        """Initialize the code generation engine with a Jinja2 environment.

        Parameters:
            templates_dir: Path to templates directory. Defaults to `templates/`
                relative to this module.
        """
        self._templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        # Cache for loaded template contents
        self._template_cache: dict[str, str] = {}

    @property
    def templates_dir(self) -> Path:
        """Get the templates directory path."""
        return self._templates_dir

    def render_string(self, template_str: str, context: dict[str, Any]) -> str:
        """Render a template string with the given context.

        Parameters:
            template_str: Jinja2 template as a string.
            context: Dictionary of variables to pass to the template.

        Returns:
            The rendered code as a string.
        """
        template = self._env.from_string(template_str)
        return template.render(**context)

    def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template file by name.

        Parameters:
            template_name: Name of the template file (without .jinja2 extension).
            context: Dictionary of variables to pass to the template.

        Returns:
            The rendered code as a string.

        Raises:
            jinja2.TemplateNotFound: If the template file does not exist.
        """
        template = self._env.get_template(f"{template_name}.jinja2")
        return template.render(**context)

    def get_template_content(self, template_name: str) -> str:
        """Get the raw content of a template file.

        Parameters:
            template_name: Name of the template file (without .jinja2 extension).

        Returns:
            The template content as a string.

        Raises:
            FileNotFoundError: If the template file does not exist.
        """
        if template_name not in self._template_cache:
            template_path = self._templates_dir / f"{template_name}.jinja2"
            self._template_cache[template_name] = template_path.read_text()
        return self._template_cache[template_name]

    def list_templates(self) -> list[str]:
        """List all available template names.

        Returns:
            List of template names (without .jinja2 extension).
        """
        return [p.stem for p in self._templates_dir.glob("*.jinja2") if p.is_file()]


# =============================================================================
# Base Code Generator
# =============================================================================


class BaseCodeGenerator(ABC):
    """Abstract base class for code generators.

    Code generators use the CodeGenerationEngine to produce code for specific
    domains (e.g., UDFs, queries, etc.). Subclasses must implement the
    `generate` method.
    """

    def __init__(self, engine: CodeGenerationEngine | None = None) -> None:
        """Initialize the generator with a code generation engine.

        Parameters:
            engine: CodeGenerationEngine instance. If None, creates a new one.
        """
        self._engine = engine or CodeGenerationEngine()

    @property
    def engine(self) -> CodeGenerationEngine:
        """Get the code generation engine."""
        return self._engine

    @abstractmethod
    def generate(self, context: Any) -> GeneratedCode:
        """Generate code from the given context.

        Parameters:
            context: Context object containing generation parameters.

        Returns:
            GeneratedCode object with the rendered code.
        """


# =============================================================================
# UDF Code Generator
# =============================================================================


class UDFCodeGenerator(BaseCodeGenerator):
    """Generator for UDF wrapper code.

    This generator creates Python and Pandas UDF wrapper functions that handle:
    - Output column renaming for Spark compatibility
    - Timezone conversion for datetime columns
    - Single vs multiple output handling
    """

    def generate(self, context: UDFContext) -> GeneratedCode:
        """Generate a Python UDF wrapper (default implementation).

        Parameters:
            context: UDFContext with generation parameters.

        Returns:
            GeneratedCode with the rendered wrapper function.
        """
        return self.generate_python_wrapper(context, rename_outputs=False)

    def generate_python_wrapper(
        self, context: UDFContext, rename_outputs: bool = False
    ) -> GeneratedCode:
        """Generate a Python UDF wrapper for single-value execution.

        The wrapper handles:
        - Timezone conversion for datetime outputs
        - Optional output renaming to dictionary format (for Spark UDFs)

        Parameters:
            context: UDFContext with generation parameters.
            rename_outputs: If True, outputs are returned as a dict with column names.

        Returns:
            GeneratedCode with the rendered wrapper function.
        """
        # Load the timezone converter template content
        timezone_converter = self._engine.get_template_content(
            "python_timezone_converter"
        )

        template_context = context.to_template_context(
            rename_outputs=rename_outputs,
            timezone_converter=timezone_converter,
        )
        code = self._engine.render_template("python_udf_wrapper", template_context)
        return GeneratedCode(code=code, template_name="python_udf_wrapper")

    def generate_pandas_wrapper(self, context: UDFContext) -> GeneratedCode:
        """Generate a Pandas UDF wrapper for vectorized execution.

        The wrapper handles:
        - Column renaming for Spark schema compatibility
        - Timezone conversion for datetime columns
        - Single column vs DataFrame output handling

        Parameters:
            context: UDFContext with generation parameters.

        Returns:
            GeneratedCode with the rendered wrapper function.
        """
        # Load the timezone converter template content
        timezone_converter = self._engine.get_template_content(
            "pandas_timezone_converter"
        )

        template_context = context.to_template_context(
            timezone_converter=timezone_converter,
        )
        code = self._engine.render_template("pandas_udf_wrapper", template_context)
        return GeneratedCode(code=code, template_name="pandas_udf_wrapper")

    def generate_aggregation_wrapper(self, context: UDFContext) -> GeneratedCode:
        """Generate an aggregation UDF wrapper for group-by operations.

        The wrapper handles aggregation functions that:
        - Take DataFrame or Series as input (grouped data)
        - Return scalar values, tuples, Series, or DataFrames
        - Automatically convert outputs to DataFrame format with proper column names

        Parameters:
            context: UDFContext with generation parameters.

        Returns:
            GeneratedCode with the rendered wrapper function.
        """
        # Load the timezone converter template content
        timezone_converter_pandas = self._engine.get_template_content(
            "pandas_timezone_converter"
        )
        timezone_converter_python = self._engine.get_template_content(
            "pandas_timezone_converter"
        )

        template_context = context.to_template_context(
            timezone_converter_pandas=timezone_converter_pandas,
            timezone_converter_python=timezone_converter_python,
        )
        code = self._engine.render_template("aggregation_udf_wrapper", template_context)
        return GeneratedCode(code=code, template_name="aggregation_udf_wrapper")


# =============================================================================
# Singleton Instance (Optional)
# =============================================================================

# Global engine instance for convenience
_default_engine: CodeGenerationEngine | None = None


def get_code_generation_engine() -> CodeGenerationEngine:
    """Get or create the default code generation engine singleton.

    Returns:
        The default CodeGenerationEngine instance.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = CodeGenerationEngine()
    return _default_engine


def get_udf_code_generator() -> UDFCodeGenerator:
    """Get a UDF code generator using the default engine.

    Returns:
        A UDFCodeGenerator instance.
    """
    return UDFCodeGenerator(get_code_generation_engine())

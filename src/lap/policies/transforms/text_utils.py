"""Text parsing utilities for CoT policy."""


class TextParser:
    """Utilities for parsing text data from various formats."""

    @staticmethod
    def decode_text(value, default: str = "") -> str:
        """Decode bytes or return string as-is.

        Args:
            value: Input value (bytes, string, or other).
            default: Default value if not decodable.

        Returns:
            Decoded string.
        """
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return default

    @staticmethod
    def parse_dataset_name(data: dict) -> str:
        """Extract dataset name from data dictionary.

        Args:
            data: Data dictionary potentially containing 'dataset_name'.

        Returns:
            Dataset name string.
        """
        return TextParser.decode_text(data.get("dataset_name"), default="")

    @staticmethod
    def parse_prompt(data: dict) -> str:
        """Parse and validate prompt from data dictionary.

        Handles special processing for r1_lite datasets.

        Args:
            data: Data dictionary containing 'prompt'.

        Returns:
            Parsed prompt string.

        Raises:
            AssertionError: If prompt is missing from data.
        """
        prompt = data.get("prompt")
        assert prompt is not None, "Prompt missing from data"

        prompt_str = TextParser.decode_text(prompt, default="")
        dataset_name = TextParser.parse_dataset_name(data)

        # Special handling for r1_lite datasets
        if "r1_lite" in dataset_name:
            prompt_str = prompt_str.split("@")[-1]

        return prompt_str

    @staticmethod
    def parse_caption(data: dict) -> str:
        """Parse caption for VQA samples.

        Args:
            data: Data dictionary potentially containing 'caption'.

        Returns:
            Caption string or empty string if not found.
        """
        caption = data.get("caption")
        if caption is None:
            return ""
        return TextParser.decode_text(caption, default="")

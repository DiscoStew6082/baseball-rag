import yaml


def parse_frontmatter(content: str) -> dict:
    """
    Extract YAML frontmatter from markdown content.

    Splits on the first '---' marker pair, parses the YAML between them
    with yaml.safe_load, and returns a dict with 'metadata' and 'body'.
    """
    if not content.startswith("---"):
        return {"metadata": {}, "body": content}

    end_marker = content.find("\n---", 3)
    if end_marker == -1:
        return {"metadata": {}, "body": content}

    yaml_text = content[4:end_marker].strip()
    metadata = yaml.safe_load(yaml_text) or {}

    body = content[end_marker + 4 :].lstrip("\n")

    return {"metadata": metadata, "body": body}

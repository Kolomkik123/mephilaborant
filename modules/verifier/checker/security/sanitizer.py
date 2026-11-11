from xml.sax.saxutils import escape

def wrap_untrusted(text: str, source: str = "user_input") -> str:
    """XML-like delimiting plus escaping. This is a prompt hygiene measure, not a security boundary."""
    safe_source = escape(source, {'"': '&quot;'})
    safe_text = escape(text or "")
    return (
        f'<untrusted_data source="{safe_source}">\n'
        f'{safe_text}\n'
        f'</untrusted_data>'
    )

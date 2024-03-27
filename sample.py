from collections.abc import Generator


def checkpoints(document: str, chunk_length) -> Generator[str, None, None]:
    """
    Enumerates progressively larger pieces of the document. So for document ABCD, returns [A, AB, ABC, ABCD]
    """
    return (document[:i + chunk_length] for i in range(0, len(document), chunk_length))


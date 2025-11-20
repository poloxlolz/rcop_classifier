from pathlib import Path

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    corpus: str
    act: str
    part_no: str
    part_hdr: str
    part_hdr_ita: str | None = None
    provision_no: str
    provision_hdr: str
    ext: str
    source: Path

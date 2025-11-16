from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    corpus: str
    act: str
    part_or_chapter: str
    part_or_chapter_heading: str
    heading: str | None = None
    section: str
    section_heading: str
    ext: str
    source: str

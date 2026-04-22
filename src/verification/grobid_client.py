
from __future__ import annotations

from pathlib import Path

import requests


class GrobidClient:

    def __init__(self, base_url: str, timeout_seconds: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Reuse TCP connections across multiple PDF submissions in the same run.
        self._session = requests.Session()

    def process_fulltext_tei(self, pdf_path: Path) -> str:
        endpoint_url = f"{self.base_url}/api/processFulltextDocument"
        with open(pdf_path, "rb") as pdf_file:
            multipart_data = {"input": (pdf_path.name, pdf_file, "application/pdf")}
            response = self._session.post(
                endpoint_url,
                files=multipart_data,
                timeout=self.timeout_seconds,
            )

        if response.status_code >= 400:
            raise RuntimeError(
                f"GROBID returned HTTP {response.status_code} for {pdf_path.name}: "
                f"{response.text[:200]}"
            )

        tei_xml = response.text
        if "<TEI" not in tei_xml:
            raise RuntimeError(
                f"GROBID response for {pdf_path.name} does not look like TEI XML "
                f"(first 200 chars: {tei_xml[:200]})"
            )

        return tei_xml

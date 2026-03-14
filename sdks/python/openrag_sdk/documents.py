"""OpenRAG SDK documents client.

This module provides the DocumentsClient class for managing document operations
including ingestion, status checking, and deletion.
"""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Union

from .exceptions import NotFoundError
from .models import DeleteDocumentResponse, IngestResponse, IngestTaskStatus

if TYPE_CHECKING:
    from .client import OpenRAGClient


class DocumentsClient:
    """Client for document operations."""

    def __init__(self, client: "OpenRAGClient"):
        self._client = client

    async def ingest(
        self,
        file_path: Union[str, Path, None] = None,
        *,
        file: BinaryIO | None = None,
        filename: str | None = None,
        wait: bool = True,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> Union[IngestResponse, IngestTaskStatus]:
        """
        Ingest a document into the knowledge base.

        Args:
            file_path: Path to the file to ingest.
            file: File-like object to ingest (alternative to file_path).
            filename: Filename to use when providing file object.
            wait: If True, poll until ingestion completes. If False, return immediately.
            poll_interval: Seconds between status checks when waiting.
            timeout: Maximum seconds to wait for completion.

        Returns:
            IngestTaskStatus with final status if wait=True.
            IngestResponse with task_id if wait=False.

        Raises:
            ValueError: If neither file_path nor file is provided, or if filename is missing.
            FileNotFoundError: If the specified file_path does not exist.
            TimeoutError: If ingestion doesn't complete within timeout.
        """
        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            with open(path, "rb") as f:
                files = {"file": (path.name, f)}
                response = await self._client._request(
                    "POST",
                    "/api/v1/documents/ingest",
                    files=files,
                )
        elif file is not None:
            if filename is None:
                raise ValueError("filename is required when providing file object")
            files = {"file": (filename, file)}
            response = await self._client._request(
                "POST",
                "/api/v1/documents/ingest",
                files=files,
            )
        else:
            raise ValueError("Either file_path or file must be provided")

        data = response.json()
        ingest_response = IngestResponse(**data)

        if not wait:
            return ingest_response

        # Poll for completion
        return await self.wait_for_task(
            ingest_response.task_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    async def get_task_status(self, task_id: str) -> IngestTaskStatus:
        """
        Get the status of an ingestion task.

        Args:
            task_id: The task ID returned from ingest().

        Returns:
            IngestTaskStatus with current task status.
        """
        response = await self._client._request(
            "GET",
            f"/api/v1/tasks/{task_id}",
        )
        data = response.json()
        return IngestTaskStatus(**data)

    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
    ) -> IngestTaskStatus:
        """
        Wait for an ingestion task to complete.

        Args:
            task_id: The task ID to wait for.
            poll_interval: Seconds between status checks.
            timeout: Maximum seconds to wait.

        Returns:
            IngestTaskStatus with final status.

        Raises:
            TimeoutError: If task doesn't complete within timeout.
        """
        start_time = time.monotonic()
        while True:
            status = await self.get_task_status(task_id)
            if status.status in ("completed", "failed"):
                return status
            
            elapsed = time.monotonic() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Ingestion task {task_id} did not complete within {timeout}s"
                )
            
            # Sleep for remaining time or poll_interval, whichever is smaller
            remaining = timeout - elapsed
            sleep_time = min(poll_interval, remaining)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def delete(self, filename: str) -> DeleteDocumentResponse:
        """
        Delete a document from the knowledge base.

        This operation is idempotent - deleting a non-existent document
        will return a success response with 0 deleted chunks instead of raising an error.

        Args:
            filename: Name of the file to delete.

        Returns:
            DeleteDocumentResponse with deleted chunk count and status.

        Note:
            If the document doesn't exist, returns a response with success=False
            and deleted_chunks=0 instead of raising an exception.
        """
        try:
            response = await self._client._request(
                "DELETE",
                "/api/v1/documents",
                json={"filename": filename},
            )
        except NotFoundError as e:
            # Keep delete idempotent for SDK callers: a missing document is not an exception.
            if getattr(e, "status_code", None) == 404:
                return DeleteDocumentResponse(
                    success=False,
                    deleted_chunks=0,
                    filename=filename,
                    message="Document not found",
                    error=getattr(e, "message", "Resource not found"),
                )
            raise

        data = response.json()
        return DeleteDocumentResponse(**data)

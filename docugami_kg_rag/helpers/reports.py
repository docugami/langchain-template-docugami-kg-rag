from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional
import requests
from docugami import Docugami

from langchain_docugami.tools.reports import (
    connect_to_db,
    excel_to_sqlite_connection,
    report_name_to_report_query_tool_function_name,
    report_details_to_report_query_tool_description,
)
from docugami_kg_rag.config import DOCUGAMI_API_KEY, INDEXING_LOCAL_REPORT_DBS_ROOT

HEADERS = {"Authorization": f"Bearer {DOCUGAMI_API_KEY}"}


@dataclass
class ReportDetails:
    id: str
    """ID of report."""

    name: str
    """Name of report."""

    local_xlsx_path: Path
    """Local path to XLSX of the report."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. sql_query_earnings_calls."""

    retrieval_tool_description: str
    """
    Description of retrieval tool e.g. Runs a SQL query over the REPORT_NAME report, 
    represented as the following SQL Table... etc."""


def download_project_latest_xlsx(project_url: str, local_xlsx: Path) -> Optional[Path]:
    response = requests.request(
        "GET",
        project_url + "/artifacts/latest?name=spreadsheet.xlsx",
        headers=HEADERS,
        data={},
    )
    if response.ok:
        response_json = response.json()["artifacts"]
        xlsx_artifact = next(
            (item for item in response_json if str(item["name"]).lower().endswith(".xlsx")),
            None,
        )
        if xlsx_artifact:
            artifact_id = xlsx_artifact["id"]
            response = requests.request(
                "GET",
                project_url + f"/artifacts/latest/{artifact_id}/content",
                headers=HEADERS,
                data={},
            )
            if response.ok:
                os.makedirs(str(local_xlsx.parent), exist_ok=True)
                with open(local_xlsx, "wb") as f:
                    f.write(response.content)
                    return local_xlsx
            else:
                raise Exception(
                    f"Failed to download XLSX for {project_url}",
                )
    elif response.status_code == 404:
        # No artifacts found: this project has never been published
        return None
    else:
        raise Exception(f"Failed to download XLSX for {project_url}")


def build_report_details(docset_id: str) -> List[ReportDetails]:
    docugami_client = Docugami()

    projects_response = docugami_client.projects.list()
    if not projects_response or not projects_response.projects:
        return []  # no projects found

    projects = [p for p in projects_response.projects if p.docset.id == docset_id]
    details: List[ReportDetails] = []
    for project in projects:
        local_xlsx_path = download_project_latest_xlsx(
            project.url, Path(INDEXING_LOCAL_REPORT_DBS_ROOT) / f"{project.id}.xlsx"
        )
        if local_xlsx_path:
            report_name = project.name or local_xlsx_path.name
            conn = excel_to_sqlite_connection(local_xlsx_path, report_name)
            db = connect_to_db(conn)
            table_info = db.get_table_info()
            details.append(
                ReportDetails(
                    id=project.id,
                    name=report_name,
                    local_xlsx_path=local_xlsx_path,
                    retrieval_tool_function_name=report_name_to_report_query_tool_function_name(project.name),
                    retrieval_tool_description=report_details_to_report_query_tool_description(project.name, table_info),
                )
            )

    return details

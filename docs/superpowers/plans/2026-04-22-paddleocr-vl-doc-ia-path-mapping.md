# PaddleOCR-VL Old-to-New Path Mapping

| Old location | Old behavior | New location | Why the new path is equivalent |
| --- | --- | --- | --- |
| Main guide `Process Guide / 流程导览` flowchart | Ask the user to decide hardware and workflow in one place | Main guide hardware entry block plus main-guide workflow block for the default path | The same decisions still exist, but they are split into smaller, more local navigation units |
| Hardware guide intro `TIP` that sends the reader back to the main guide process guide | Force a round trip to choose a workflow | Local workflow guide block inside the same hardware guide | The hardware decision is already complete; the workflow decision becomes local and clearer |
| Main guide flowchart branch `Need to confirm supported inference methods first?` | Make support-matrix reading part of the global route | Optional note below each entry block pointing to the support matrix only when needed | The support matrix remains available, but it stops acting as a mandatory route branch |
| Apple Silicon service deployment note | Manual-only support is implied after the reader reaches Section 4 | Apple workflow guide row marks full API service as manual-only before the reader reaches Section 4 | The support fact stays the same, but the reader sees it earlier |
| Compose-only hardware guides service deployment sections | Full API support is discoverable only after entering Section 4 | Each compose-only hardware guide exposes full API service support in the local workflow table | The implementation body stays local; only the route becomes explicit |
| Blackwell manual deployment subsection | Manual deployment is available, but the routing decision is still upstream in the main flowchart | Blackwell local workflow guide explicitly advertises both Docker Compose and manual deployment | The route stays accurate without sending the reader back to the main guide |

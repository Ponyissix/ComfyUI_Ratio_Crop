from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import server
from aiohttp import web
import os
import folder_paths

# --- Add Custom API Route for Deletion ---
@server.PromptServer.instance.routes.post("/ratio-crop/delete")
async def delete_file(request):
    try:
        data = await request.json()
        filename = data.get("filename")
        subfolder = data.get("subfolder", "")
        
        if not filename:
            return web.Response(status=400, text="Filename is required")

        # Security check: Ensure we are only deleting from the input directory
        input_dir = folder_paths.get_input_directory()
        
        # Construct full path
        if subfolder:
            target_path = os.path.join(input_dir, subfolder, filename)
            # Ensure subfolder stays within input_dir (prevent directory traversal)
            parent_dir = os.path.abspath(os.path.join(input_dir, subfolder))
        else:
            target_path = os.path.join(input_dir, filename)
            parent_dir = os.path.abspath(input_dir)

        # Canonicalize paths to check containment
        target_path = os.path.abspath(target_path)
        common_prefix = os.path.commonpath([target_path, input_dir])
        
        # Verify that the target path is indeed inside the input directory
        if common_prefix != input_dir:
             return web.Response(status=403, text="Access denied: Cannot delete files outside input directory")
        
        if os.path.exists(target_path) and os.path.isfile(target_path):
            os.remove(target_path)
            print(f"[RatioCrop] Deleted file: {target_path}")
            return web.json_response({"status": "success"})
        else:
            return web.Response(status=404, text="File not found")

    except Exception as e:
        print(f"[RatioCrop] Delete failed: {e}")
        return web.Response(status=500, text=str(e))

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

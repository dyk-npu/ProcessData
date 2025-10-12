import argparse
import pathlib

import dgl
import numpy as np
import torch
from occwl.graph import face_adjacency
from occwl.io import load_step
from occwl.shell import Shell
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from occwl.uvgrid import ugrid, uvgrid
from tqdm import tqdm
from multiprocessing.pool import Pool
from itertools import repeat
import signal
from multiprocessing import Manager


def load_stl_file(file_path):
    """Load STL file and convert to solid"""
    try:
        stl_reader = StlAPI_Reader()
        shape = TopoDS_Shape()

        if not stl_reader.Read(shape, str(file_path)):
            return None

        # Try to sew the mesh into a solid
        sewing = BRepBuilderAPI_Sewing()
        sewing.Add(shape)
        sewing.Perform()

        # Get the sewn shape
        if hasattr(sewing, 'IsDone') and sewing.IsDone():
            sewn_shape = sewing.SewedShape()
            if sewn_shape and not sewn_shape.IsNull():
                return Shell(sewn_shape)
        else:
            # Try to get sewn shape directly
            try:
                sewn_shape = sewing.SewedShape()
                if sewn_shape and not sewn_shape.IsNull():
                    return Shell(sewn_shape)
            except:
                pass

        # Return original shape if sewing fails
        if not shape.IsNull():
            # Convert TopoDS_Shape to occwl Shell
            return Shell(shape)
        return None

    except Exception as e:
        print(f"Error loading STL file {file_path}: {e}")
        return None


def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def process_one_file(arguments):
    fn, args, stats = arguments
    fn_stem = fn.stem
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Check file size and skip if too large
    file_size_kb = fn.stat().st_size / 1024
    if file_size_kb > args.max_file_size:
        with stats['lock']:
            stats['skipped'] += 1
        return {'status': 'skipped', 'file': str(fn), 'output': '', 'reason': f'File too large: {file_size_kb:.1f} KB (limit: {args.max_file_size} KB)'}

    # Preserve directory structure when using recursive mode
    if args.recursive:
        # Get relative path from input root
        rel_path = fn.relative_to(input_path)
        # Create corresponding output subdirectory
        output_subdir = output_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / (fn_stem + ".bin")
    else:
        output_file = output_path / (fn_stem + ".bin")

    # Check if output file already exists and skip if requested
    if args.skip_existing and output_file.exists():
        with stats['lock']:
            stats['skipped'] += 1
        return {'status': 'skipped', 'file': str(fn), 'output': str(output_file), 'reason': 'Output file already exists'}

    try:
        # Determine file type and load accordingly
        file_ext = fn.suffix.lower()

        if file_ext in ['.stl']:
            # Load STL file
            solid = load_stl_file(fn)
            if solid is None:
                with stats['lock']:
                    stats['failed'] += 1
                return {'status': 'error', 'file': str(fn), 'output': str(output_file), 'error': 'Failed to load STL file - returned None'}
        elif file_ext in ['.step', '.stp']:
            # Load STEP file and check if it contains any solids
            solids = load_step(fn)
            if not solids or len(solids) == 0:
                with stats['lock']:
                    stats['failed'] += 1
                return {'status': 'error', 'file': str(fn), 'output': str(output_file), 'error': 'No solids found in STEP file'}
            solid = solids[0]
        else:
            with stats['lock']:
                stats['failed'] += 1
            return {'status': 'error', 'file': str(fn), 'output': str(output_file), 'error': f'Unsupported file format: {file_ext}'}

        # Build graph
        graph = build_graph(
            solid, args.curv_u_samples, args.surf_u_samples, args.surf_v_samples
        )

        # Save graph
        dgl.data.utils.save_graphs(str(output_file), [graph])

        with stats['lock']:
            stats['processed'] += 1
        return {'status': 'success', 'file': str(fn), 'output': str(output_file)}

    except Exception as e:
        with stats['lock']:
            stats['failed'] += 1
        return {'status': 'error', 'file': str(fn), 'output': str(output_file), 'error': str(e)}


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process(args):
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Search for STEP and STL files
    if args.recursive:
        step_files = list(input_path.rglob("*.st*p")) + list(input_path.rglob("*.stl"))
        print(f"Searching recursively in {input_path}")
    else:
        step_files = list(input_path.glob("*.st*p")) + list(input_path.glob("*.stl"))
        print(f"Searching in {input_path} (non-recursive)")

    if not step_files:
        print(f"No STEP/STL files found in {input_path}")
        return

    print(f"Found {len(step_files)} STEP/STL files")
    print(f"Using {args.num_processes} processes")
    if args.skip_existing:
        print("Skip existing: Enabled")
    else:
        print("Skip existing: Disabled")

    # Create shared statistics
    manager = Manager()
    stats = manager.dict()
    stats['processed'] = 0
    stats['skipped'] = 0
    stats['failed'] = 0
    stats['lock'] = manager.Lock()

    # for fn in tqdm(step_files):
    #     process_one_file(fn, args)
    pool = Pool(processes=args.num_processes, initializer=initializer)
    try:
        results = list(tqdm(
            pool.imap(process_one_file, zip(step_files, repeat(args), repeat(stats))),
            total=len(step_files),
            desc="Processing"
        ))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print("\nProcess interrupted by user")
        return

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("🎉 Processing completed!")
    print(f"✅ Successfully processed: {stats['processed']} files")
    print(f"⏭️  Skipped existing: {stats['skipped']} files")
    print(f"❌ Failed: {stats['failed']} files")
    print(f"📊 Total files: {len(step_files)}")

    if len(step_files) > 0:
        success_rate = (stats['processed'] / len(step_files)) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")

    # Show skipped files breakdown
    skipped_results = [r for r in results if r and r.get('status') == 'skipped']
    if skipped_results:
        # Count skip reasons
        skip_reasons = {}
        for result in skipped_results:
            reason = result.get('reason', 'Unknown reason')
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        print("\nSkipped files breakdown:")
        for reason, count in skip_reasons.items():
            print(f"  {reason}: {count} files")

    # Show failed files if any
    failed_results = [r for r in results if r and r.get('status') == 'error']
    if failed_results:
        print("\nFailed files:")
        for i, result in enumerate(failed_results[:10], 1):
            print(f"  {i}. {pathlib.Path(result['file']).name}: {result['error']}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more files")


def main():
    parser = argparse.ArgumentParser(
        "Convert solid models to face-adjacency graphs with UV-grid features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process STEP files in a single directory
  python brep_preprocessor.py ./step_files ./output_graphs

  # Process STL files recursively with custom sampling
  python brep_preprocessor.py ./stl_files ./output_graphs -r --surf_u_samples 20 --surf_v_samples 20

  # Use 16 processes and skip existing files
  python brep_preprocessor.py ./cad_files ./output_graphs --num_processes 16 --skip-existing

  # Process recursively and skip existing (supports both STEP and STL)
  python brep_preprocessor.py ./cad_files ./output_graphs -r --skip-existing

  # Skip files larger than 2MB
  python brep_preprocessor.py ./cad_files ./output_graphs --max-file-size 2048
        """
    )
    parser.add_argument("input", type=str, help="Input folder of STEP/STL files")
    parser.add_argument("output", type=str, help="Output folder of DGL graph BIN files")
    parser.add_argument(
        "--curv_u_samples", type=int, default=10, help="Number of samples on each curve"
    )
    parser.add_argument(
        "--surf_u_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the u-direction",
    )
    parser.add_argument(
        "--surf_v_samples",
        type=int,
        default=10,
        help="Number of samples on each surface along the v-direction",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=12,
        help="Number of processes to use",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for STEP files recursively in subdirectories",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output .bin file already exists",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1500,
        help="Maximum file size in KB to process (default: 1500)",
    )
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
"""
3D Printable Raspberry Pi Enclosure Generator
Generates STL files for 3D printing

Specifications:
- Main box: 150mm x 150mm x 150mm
- Base thickness: 6mm
- Raspberry Pi compartment: 100mm x 70mm x 50mm
- Side hole for power cable
- Removable top with camera hole
- GPS antenna mount on top
- All measurements in millimeters
"""

import numpy as np
from stl import mesh

def create_box_with_cavity(outer_size, wall_thickness, cavity_size, cavity_offset):
    """
    Create a box with a cavity inside
    
    Args:
        outer_size: (x, y, z) outer dimensions
        wall_thickness: thickness of walls
        cavity_size: (x, y, z) cavity dimensions
        cavity_offset: (x, y, z) offset from corner to cavity
    """
    vertices = []
    faces = []
    
    ox, oy, oz = outer_size
    cx, cy, cz = cavity_size
    offx, offy, offz = cavity_offset
    
    # Outer box vertices (8 corners)
    outer_verts = [
        [0, 0, 0],      # 0
        [ox, 0, 0],     # 1
        [ox, oy, 0],    # 2
        [0, oy, 0],     # 3
        [0, 0, oz],     # 4
        [ox, 0, oz],    # 5
        [ox, oy, oz],   # 6
        [0, oy, oz]     # 7
    ]
    
    # Inner cavity vertices (8 corners)
    inner_verts = [
        [offx, offy, wall_thickness],                    # 8
        [offx + cx, offy, wall_thickness],               # 9
        [offx + cx, offy + cy, wall_thickness],          # 10
        [offx, offy + cy, wall_thickness],               # 11
        [offx, offy, wall_thickness + cz],               # 12
        [offx + cx, offy, wall_thickness + cz],          # 13
        [offx + cx, offy + cy, wall_thickness + cz],     # 14
        [offx, offy + cy, wall_thickness + cz]           # 15
    ]
    
    vertices = outer_verts + inner_verts
    
    # Outer box faces
    faces = [
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top (will be open)
        # [4, 5, 6], [4, 6, 7],
        # Sides
        [0, 1, 5], [0, 5, 4],  # Front
        [1, 2, 6], [1, 6, 5],  # Right
        [2, 3, 7], [2, 7, 6],  # Back
        [3, 0, 4], [3, 4, 7],  # Left
    ]
    
    # Inner cavity faces (inverted normals)
    cavity_faces = [
        # Bottom of cavity
        [8, 9, 10], [8, 10, 11],
        # Sides of cavity
        [8, 12, 9], [9, 12, 13],  # Front
        [9, 13, 10], [10, 13, 14],  # Right
        [10, 14, 11], [11, 14, 15],  # Back
        [11, 15, 8], [8, 15, 12],  # Left
    ]
    
    faces.extend(cavity_faces)
    
    # Connect outer to inner (walls)
    # Front wall
    faces.extend([
        [0, 8, 11], [0, 11, 3],
        [1, 9, 8], [1, 8, 0],
        [1, 2, 10], [1, 10, 9],
        [2, 3, 11], [2, 11, 10]
    ])
    
    # Side walls from cavity top to box top
    faces.extend([
        [12, 4, 7], [12, 7, 15],
        [13, 5, 4], [13, 4, 12],
        [14, 6, 5], [14, 5, 13],
        [15, 7, 6], [15, 6, 14]
    ])
    
    return np.array(vertices), np.array(faces)


def create_main_box():
    """Create the main enclosure box"""
    print("Creating main box...")
    
    outer_size = (150, 150, 150)  # 15cm cube
    wall_thickness = 6  # 6mm base
    cavity_size = (100, 70, 50)  # Raspberry Pi compartment
    cavity_offset = (25, 40, 0)  # Center the cavity
    
    vertices, faces = create_box_with_cavity(outer_size, wall_thickness, 
                                             cavity_size, cavity_offset)
    
    # Create mesh
    box_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            box_mesh.vectors[i][j] = vertices[face[j]]
    
    box_mesh.save('raspi_enclosure_main.stl')
    print("✅ Created: raspi_enclosure_main.stl")
    return box_mesh


def create_removable_top():
    """Create removable top with camera hole"""
    print("\nCreating removable top...")
    
    # Top plate dimensions
    width = 148  # Slightly smaller than outer box
    depth = 148
    thickness = 4
    
    # Camera hole
    camera_hole_x = 75
    camera_hole_y = 75
    camera_hole_size = 30
    
    vertices = []
    faces = []
    
    # Create top plate with hole
    # Outer corners
    v0 = [0, 0, 0]
    v1 = [width, 0, 0]
    v2 = [width, depth, 0]
    v3 = [0, depth, 0]
    v4 = [0, 0, thickness]
    v5 = [width, 0, thickness]
    v6 = [width, depth, thickness]
    v7 = [0, depth, thickness]
    
    # Camera hole corners (on top surface)
    cx1 = camera_hole_x - camera_hole_size/2
    cx2 = camera_hole_x + camera_hole_size/2
    cy1 = camera_hole_y - camera_hole_size/2
    cy2 = camera_hole_y + camera_hole_size/2
    
    v8 = [cx1, cy1, thickness]
    v9 = [cx2, cy1, thickness]
    v10 = [cx2, cy2, thickness]
    v11 = [cx1, cy2, thickness]
    
    # Camera hole bottom
    v12 = [cx1, cy1, 0]
    v13 = [cx2, cy1, 0]
    v14 = [cx2, cy2, 0]
    v15 = [cx1, cy2, 0]
    
    vertices = [v0, v1, v2, v3, v4, v5, v6, v7, 
                v8, v9, v10, v11, v12, v13, v14, v15]
    
    # Bottom face (with hole)
    faces = [
        [0, 1, 13], [0, 13, 12],
        [1, 2, 14], [1, 14, 13],
        [2, 3, 15], [2, 15, 14],
        [3, 0, 12], [3, 12, 15]
    ]
    
    # Top face (with hole) - create ring
    faces.extend([
        # Top surface sections around hole
        [4, 8, 11], [4, 11, 7],
        [4, 5, 9], [4, 9, 8],
        [5, 6, 10], [5, 10, 9],
        [6, 7, 11], [6, 11, 10]
    ])
    
    # Outer walls
    faces.extend([
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0]
    ])
    
    # Camera hole walls
    faces.extend([
        [8, 12, 13], [8, 13, 9],
        [9, 13, 14], [9, 14, 10],
        [10, 14, 15], [10, 15, 11],
        [11, 15, 12], [11, 12, 8]
    ])
    
    top_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    vertices_array = np.array(vertices)
    for i, face in enumerate(faces):
        for j in range(3):
            top_mesh.vectors[i][j] = vertices_array[face[j]]
    
    top_mesh.save('raspi_enclosure_top.stl')
    print("✅ Created: raspi_enclosure_top.stl")
    return top_mesh


def create_gps_mount():
    """Create GPS antenna mount - attaches to top edge (not removable)"""
    print("\nCreating GPS mount...")
    
    # GPS mount plate - larger to attach to box edge
    width = 50
    depth = 50
    thickness = 3
    
    # Mount post
    post_height = 15  # Taller for better GPS clearance
    post_width = 35
    
    vertices = []
    faces = []
    
    # Base plate
    v0 = [0, 0, 0]
    v1 = [width, 0, 0]
    v2 = [width, depth, 0]
    v3 = [0, depth, 0]
    v4 = [0, 0, thickness]
    v5 = [width, 0, thickness]
    v6 = [width, depth, thickness]
    v7 = [0, depth, thickness]
    
    # Post
    px1 = (width - post_width) / 2
    px2 = px1 + post_width
    py1 = (depth - post_width) / 2
    py2 = py1 + post_width
    pz = thickness + post_height
    
    v8 = [px1, py1, thickness]
    v9 = [px2, py1, thickness]
    v10 = [px2, py2, thickness]
    v11 = [px1, py2, thickness]
    v12 = [px1, py1, pz]
    v13 = [px2, py1, pz]
    v14 = [px2, py2, pz]
    v15 = [px1, py2, pz]
    
    vertices = [v0, v1, v2, v3, v4, v5, v6, v7,
                v8, v9, v10, v11, v12, v13, v14, v15]
    
    # Base faces
    faces = [
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Sides
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ]
    
    # Post faces
    faces.extend([
        # Top
        [12, 13, 14], [12, 14, 15],
        # Sides
        [8, 9, 13], [8, 13, 12],
        [9, 10, 14], [9, 14, 13],
        [10, 11, 15], [10, 15, 14],
        [11, 8, 12], [11, 12, 15]
    ])
    
    gps_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    vertices_array = np.array(vertices)
    for i, face in enumerate(faces):
        for j in range(3):
            gps_mesh.vectors[i][j] = vertices_array[face[j]]
    
    gps_mesh.save('raspi_enclosure_gps_mount.stl')
    print("✅ Created: raspi_enclosure_gps_mount.stl")
    return gps_mesh


def create_power_hole_template():
    """Create template for power cable hole (to be cut manually or with modifier)"""
    print("\nCreating power hole template...")
    
    # Simple cylinder for the hole
    radius = 6  # 12mm diameter hole
    height = 10  # Wall thickness
    segments = 20
    
    vertices = []
    faces = []
    
    # Bottom circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, 0])
    
    # Top circle
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, height])
    
    # Bottom cap
    for i in range(1, segments - 1):
        faces.append([0, i + 1, i])
    
    # Top cap
    for i in range(1, segments - 1):
        faces.append([segments, segments + i, segments + i + 1])
    
    # Side faces
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([i, segments + i, next_i])
        faces.append([next_i, segments + i, segments + next_i])
    
    hole_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    vertices_array = np.array(vertices)
    for i, face in enumerate(faces):
        for j in range(3):
            hole_mesh.vectors[i][j] = vertices_array[face[j]]
    
    hole_mesh.save('raspi_enclosure_power_hole.stl')
    print("✅ Created: raspi_enclosure_power_hole.stl (Use for reference)")
    return hole_mesh


def main():
    """Generate all STL files"""
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "3D PRINTABLE ENCLOSURE GENERATOR" + " "*16 + "║")
    print("╚" + "="*58 + "╝\n")
    
    try:
        import numpy as np
        from stl import mesh as stl_mesh
        print("✅ Dependencies loaded\n")
    except ImportError:
        print("❌ Error: numpy-stl not installed")
        print("   Install with: pip install numpy-stl\n")
        return
    
    print("📏 Specifications:")
    print("   Main box: 150mm x 150mm x 150mm")
    print("   Base thickness: 6mm")
    print("   Raspberry Pi cavity: 100mm x 70mm x 50mm")
    print("   Wall thickness: 6mm")
    print("   Camera hole: 30mm x 30mm")
    print("   GPS mount: 40mm x 40mm x 13mm\n")
    
    print("-" * 60)
    
    # Generate all parts
    create_main_box()
    create_removable_top()
    create_gps_mount()
    create_power_hole_template()
    
    print("\n" + "=" * 60)
    print("   ✅ ALL STL FILES GENERATED!")
    print("=" * 60)
    print("\n📁 Files created:")
    print("   1. raspi_enclosure_main.stl      - Main box with cavity")
    print("   2. raspi_enclosure_top.stl       - Removable top with camera hole")
    print("   3. raspi_enclosure_gps_mount.stl - GPS mount (glue to top edge)")
    print("   4. raspi_enclosure_power_hole.stl - Power cable hole template")
    
    print("\n📝 Instructions for 3D printing shop:")
    print("   • Material: PLA or PETG")
    print("   • Layer height: 0.2mm")
    print("   • Infill: 20-30%")
    print("   • Supports: Yes (for top piece)")
    print("   • Print main box upside down")
    print("   • Top piece can be printed as-is")
    
    print("\n🔧 Assembly (matches your sketch):")
    print("   1. Print all 4 parts")
    print("   2. Drill power hole on SIDE near bottom (12mm)")
    print("   3. Drill GPS wire holes on TOP EDGE (3-4mm)")
    print("   4. GLUE GPS mount to top edge (NOT removable)")
    print("   5. Place Raspberry Pi in cavity")
    print("   6. Run GPS wire through holes in top edge")
    print("   7. Close with removable top (camera hole on top)\n")


if __name__ == "__main__":
    main()

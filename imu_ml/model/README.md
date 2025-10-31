# 3D Printable Raspberry Pi Enclosure

Custom enclosure for Raspberry Pi with IMU, Camera, and GPS modules.

---

## 📏 Specifications

- **Main Box**: 150mm x 150mm x 150mm cube
- **Base Thickness**: 6mm (strong base)
- **Wall Thickness**: 6mm
- **Raspberry Pi Cavity**: 100mm x 70mm x 50mm
- **Camera Hole**: 30mm x 30mm (on removable top)
- **Power Cable Hole**: 12mm diameter (side)
- **GPS Mount**: 40mm x 40mm x 13mm

---

## 🚀 How to Generate STL Files

### Step 1: Install Dependencies
```bash
pip install numpy-stl
```

### Step 2: Generate Files
```bash
cd model
python generate_enclosure.py
```

### Output Files:
1. **raspi_enclosure_main.stl** - Main box with Raspberry Pi cavity
2. **raspi_enclosure_top.stl** - Removable top with camera hole
3. **raspi_enclosure_gps_mount.stl** - GPS antenna mount
4. **raspi_enclosure_power_hole.stl** - Power cable hole template

---

## 📦 For 3D Printing Shop

### Print Settings:
```
Material: PLA or PETG
Layer Height: 0.2mm
Infill: 20-30%
Supports: Yes (for removable top)
Print Temperature: 200-220°C (PLA) or 230-250°C (PETG)
Bed Temperature: 60°C (PLA) or 80°C (PETG)
```

### Print Orientation:
- **Main Box**: Print upside down (open side on bed)
- **Top Piece**: Print as-is (flat side down)
- **GPS Mount**: Print flat side down
- **Power Hole**: Reference only (drill manually)

---

##Design Details

### Main Box (`raspi_enclosure_main.stl`)
```
┌──────────────────────────────────────┐
│ 150mm x 150mm x 150mm                │
│                                      │
│  ┌──────────────────────┐           │
│  │                      │           │
│  │  Raspberry Pi        │           │
│  │  Cavity              │           │
│  │  100mm x 70mm x 50mm │           │
│  │                      │           │
│  └──────────────────────┘           │
│                                      │
│  Base: 6mm thick                     │
└──────────────────────────────────────┘
  ○ Power hole on side (12mm)
```

### Removable Top (`raspi_enclosure_top.stl`)
```
┌──────────────────────────────────────┐
│ 148mm x 148mm x 4mm                  │
│                                      │
│           ┌──────┐                   │
│           │Camera│                   │
│           │ Hole │  ← 30mm x 30mm    │
│           └──────┘                   │
│                                      │
│                      [GPS Mount Area]│
└──────────────────────────────────────┘
```

### GPS Mount (`raspi_enclosure_gps_mount.stl`)
```
    ┌─────────┐
    │  30x30  │ ← Post (10mm high)
    │         │
┌───┴─────────┴───┐
│   40mm x 40mm   │ ← Base plate (3mm thick)
└─────────────────┘
```

---

## 🔧 Assembly Instructions

### 1. After Printing

**Main Box:**
- Remove supports if any
- Drill 12mm hole on side wall for power cable
  - Location: 20mm from bottom, center of wall
  - Use power hole template for reference

**Top Piece:**
- Remove supports around camera hole
- Ensure camera can fit through hole (test with actual camera)

**GPS Mount:**
- Clean any stringing
- Test fit on top edge

### 2. Hardware Installation

**Install in this order:**
1. Place Raspberry Pi in cavity (100x70x50mm space)
2. Connect power cable through side hole
3. Mount IMU sensor inside
4. Attach GPS mount to top edge
5. Position GPS antenna on mount
6. Place camera to align with top hole
7. Close with removable top

### 3. Securing Components

**Options:**
- Use hot glue for permanent mounting
- Use double-sided tape for temporary mounting
- Add screw holes if needed (drill with 3mm bit)

---

## 📐 Dimensions Reference

### Main Enclosure
- Outer: 150 x 150 x 150 mm
- Inner cavity: 100 x 70 x 50 mm
- Wall thickness: 6 mm all around
- Base thickness: 6 mm

### Openings
- Camera hole: 30 x 30 mm (square)
- Power hole: 12 mm diameter (drill manually)
- Top opening: 148 x 148 mm

### GPS Mount
- Base: 40 x 40 x 3 mm
- Post: 30 x 30 x 10 mm
- Total height: 13 mm

---

## 💡 Modifications

### To Resize:
Edit `generate_enclosure.py`:
```python
# Line 111-114
outer_size = (150, 150, 150)  # Change main box size
cavity_size = (100, 70, 50)   # Change Raspberry Pi space
```

### To Add Ventilation Holes:
Use your slicer software to add small holes on sides.

### To Add Mounting Brackets:
Edit the generate script to add mounting tabs.

---

## 🎨 Customization Ideas

1. **Add Logo**: Use slicer's text feature to add logo on top
2. **Color**: Print different parts in different colors
3. **Transparent Top**: Use transparent PETG for status LED visibility
4. **Cooling**: Add fan mount holes (40mm or 50mm fan)

---

## 📋 Material Recommendations

### PLA (Recommended for indoor use)
- ✅ Easy to print
- ✅ Good detail
- ✅ Cheap
- ❌ Not weatherproof
- ❌ Can warp in heat

### PETG (Recommended for outdoor use)
- ✅ Weather resistant
- ✅ Stronger than PLA
- ✅ Heat resistant
- ❌ Slightly harder to print
- ❌ More expensive

### ABS (For maximum durability)
- ✅ Very strong
- ✅ Heat resistant
- ✅ Can be smoothed with acetone
- ❌ Requires heated enclosure
- ❌ Warps easily

---

## ⚠️ Important Notes

1. **Test Fit**: Always do a test print of main box to verify Raspberry Pi fits
2. **Cooling**: Consider ventilation holes if components run hot
3. **Wire Management**: Plan cable routes before final assembly
4. **Camera Focus**: Ensure camera hole position matches camera module
5. **GPS Clearance**: Make sure GPS has clear view of sky

---

## 📸 Assembly Checklist

- [ ] All 4 STL files generated
- [ ] Files sent to 3D printing shop or loaded in slicer
- [ ] Print settings configured
- [ ] Supports enabled for top piece
- [ ] Correct orientation set
- [ ] Print completed and cleaned
- [ ] Power hole drilled
- [ ] Raspberry Pi fits in cavity
- [ ] Camera aligns with hole
- [ ] GPS mount attached
- [ ] All components fit
- [ ] Top closes properly

---

## 🛠️ If Print Fails

### Top piece doesn't fit:
- Scale down to 98% in slicer
- Sand edges if too tight

### Raspberry Pi doesn't fit:
- Regenerate with larger cavity (105x75x55mm)
- Sand cavity if close

### Camera hole too small:
- Drill/file to enlarge
- Or regenerate with larger hole (35x35mm)

---

## 📞 Support

For issues with:
- **STL generation**: Check Python script and dependencies
- **3D printing**: Consult your printer manual or printing shop
- **Assembly**: Refer to assembly instructions above

---

**Ready to print! Take the STL files to your 3D printing shop! 🖨️**

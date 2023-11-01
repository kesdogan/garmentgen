import svgpathtools as svgpath
from copy import deepcopy
import numpy as np

# Custom
import pypattern as pyp

# other assets
from . import bands


# TODO Cuffs
# FIXME Slides too high

class PantPanel(pyp.Panel):
    def __init__(
            self, name, body, design, 
            waist, hips,
            double_dart=False) -> None:
        """
            Basic pant panel with option to be fitted (with darts) or ruffled at waist area.
        """
        super().__init__(name)

        # FIXME Fix pant width parameter to change appropriately in asymmetric pants
        pant_width = design['width']['v'] * hips 
        length = design['length']['v'] * body['leg_length']
        flare = design['flare']['v'] 
        low_width = design['width']['v'] * body['hips'] * (flare - 1) / 4  + hips

        hips_depth = body['hips_line']
        hip_side_incl = np.deg2rad(body['hip_inclination'])
        dart_position = body['bust_points'] / 2
        dart_depth = hips_depth * 0.8  # FIXME check

        # Crotch cotrols
        crotch_depth_diff =  body['crotch_hip_diff']
        crotch_extention = body['leg_circ'] / 2 - body['hips'] / 4

        # eval pants shape
        # TODO Return ruffle opportunity?

        # amount of extra fabric at waist
        w_diff = pant_width - waist   # Assume its positive since waist is smaller then hips
        # We distribute w_diff among the side angle and a dart 
        hw_shift = np.tan(hip_side_incl) * hips_depth
        # Small difference
        if hw_shift > w_diff:
            hw_shift = w_diff

        # --- Edges definition ---
        # Right
        if pyp.close_enough(flare, 1):  # skip optimization
            right_bottom = pyp.Edge(    
                [hips - low_width, 0], 
                [0, length]
            )
        else:
            right_bottom = pyp.esf.curve_from_tangents(
                [hips - low_width, 0], 
                [0, length],
                target_tan1=np.array([0, 1]), 
                # initial guess places control point closer to the hips 
                initial_guess=[0.75, 0]
            )
        right_top = pyp.esf.curve_from_tangents(
            right_bottom.end,    
            [hw_shift, length + hips_depth],
            target_tan0=np.array([0, 1]),
            initial_guess=[0.5, 0] 
        )
       
        top = pyp.Edge(
            right_top.end, 
            [w_diff + waist, length + hips_depth] 
        )

        crotch = pyp.CurveEdge(
            top.end,
            [pant_width + crotch_extention, length - crotch_depth_diff], 
            [[0.9, -0.3]]    # NOTE: relative contols allow adaptation to different bodies
        )

        # Apply the rise
        # NOTE applying rise here for correctly collecting the edges
        rise = design['rise']['v']
        if not pyp.utils.close_enough(rise, 1.):
            new_level = top.end[1] - (1 - rise) * hips_depth
            right_top, top, crotch = self.apply_rise(new_level, right_top, top, crotch)

        # TODO With target tangent!
        left = pyp.CurveEdge(
            crotch.end,
            [
                min(pant_width, pant_width - (pant_width - low_width) / 2), 
                min(0, length - crotch_depth_diff)], 
            [[0.2, -0.1]]
        )

        self.edges = pyp.EdgeSequence(right_bottom, right_top, top, crotch, left).close_loop()
        bottom = self.edges[-1]

        # Default placement
        self.set_pivot(crotch.end)
        self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0] 

        # Out interfaces (easier to define before adding a dart)
        self.interfaces = {
            'outside': pyp.Interface(self, pyp.EdgeSequence(right_bottom, right_top)),
            'crotch': pyp.Interface(self, crotch),
            'inside': pyp.Interface(self, left),
            'bottom': pyp.Interface(self, bottom)
        }

        # Add top dart 
        dart_width = w_diff - hw_shift
        if w_diff > hw_shift:
            top_edges, int_edges = self.add_darts(
                top, dart_width, dart_depth, dart_position, double_dart=double_dart)
            self.interfaces['top'] = pyp.Interface(self, int_edges) 
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces['top'] = pyp.Interface(self, top) 

    def apply_rise(self, level, right, top, crotch):

        # TODOLOW This could be an operator or edge function
        right_c, crotch_c = right.as_curve(), crotch.as_curve()
        cutout = svgpath.Line(0 + 1j*level, crotch.end[0] + 1j*level)

        right_intersect = right_c.intersect(cutout)[0]
        right_cut = right_c.cropped(0, right_intersect[0])
        new_right = pyp.CurveEdge.from_svg_curve(right_cut)

        c_intersect = crotch_c.intersect(cutout)[0]
        c_cut = crotch_c.cropped(c_intersect[0], 1)
        new_crotch = pyp.CurveEdge.from_svg_curve(c_cut)

        new_top = pyp.Edge(new_right.end, new_crotch.start)

        return new_right, new_top, new_crotch


    def add_darts(self, top, dart_width, dart_depth, dart_position, double_dart=False):
        
        if double_dart:
            # TODOLOW Avoid hardcoding for matching with the top?
            dist = dart_position * 0.5  # Dist between darts -> dist between centers
            offsets_mid = [
                - (dart_position + dist / 2 + dart_width / 2) - dart_width / 4,   
                - (dart_position - dist / 2) - dart_width / 4,
            ]

            darts = [
                pyp.esf.dart_shape(dart_width / 2, dart_depth * 0.9), # smaller
                pyp.esf.dart_shape(dart_width / 2, dart_depth)  
            ]
        else:
            offsets_mid = [
                - dart_position - dart_width / 2,
            ]
            darts = [
                pyp.esf.dart_shape(dart_width, dart_depth)
            ]
        top_edges, int_edges = pyp.EdgeSequence(top), pyp.EdgeSequence(top)

        for off, dart in zip(offsets_mid, darts):
            left_edge_len = top_edges[-1].length()
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=left_edge_len + off,
                edge_seq=top_edges, 
                int_edge_seq=int_edges
            )

        return top_edges, int_edges
        

class PantsHalf(pyp.Component):
    def __init__(self, tag, body, design) -> None:
        super().__init__(tag)
        design = design['pants']

        self.front = PantPanel(
            f'pant_f_{tag}', body, design,
            waist=(body['waist'] - body['waist_back_width']) / 2,
            hips=(body['hips'] - body['hip_back_width']) / 2,
            ).translate_by([0, body['waist_level'] - 5, 25])
        self.back = PantPanel(
            f'pant_b_{tag}', body, design,
            waist=body['waist_back_width'] / 2,
            hips=body['hip_back_width'] / 2,
            double_dart=True
            ).translate_by([0, body['waist_level'] - 5, -20])

        self.stitching_rules = pyp.Stitches(
            (self.front.interfaces['outside'], self.back.interfaces['outside']),
            (self.front.interfaces['inside'], self.back.interfaces['inside'])
        )

        # add a cuff
        # TODOLOW This process is the same for sleeves -- make a function?
        if design['cuff']['type']['v']:
            
            pant_bottom = pyp.Interface.from_multiple(
                    self.front.interfaces['bottom'], self.back.interfaces['bottom'])

            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cdesign['cuff']['b_width'] = {}
            cdesign['cuff']['b_width']['v'] = pant_bottom.edges.length() / design['cuff']['top_ruffle']['v']

            # Init
            cuff_class = getattr(bands, cdesign['cuff']['type']['v'])
            self.cuff = cuff_class(tag, cdesign)

            # Position
            self.cuff.place_by_interface(
                self.cuff.interfaces['top'],
                pant_bottom,
                gap=5
            )

            # Stitch
            self.stitching_rules.append((
                pant_bottom,
                self.cuff.interfaces['top'])
            )

        self.interfaces = {
            'crotch_f': self.front.interfaces['crotch'],
            'crotch_b': self.back.interfaces['crotch'],
            'top_f': self.front.interfaces['top'],
            'top_b': self.back.interfaces['top'],
        }

class Pants(pyp.Component):
    def __init__(self, body, design) -> None:
        super().__init__('Pants')


        self.right = PantsHalf('r', body, design)
        self.left = PantsHalf('l', body, design).mirror()

        self.stitching_rules = pyp.Stitches(
            (self.right.interfaces['crotch_f'], self.left.interfaces['crotch_f']),
            (self.right.interfaces['crotch_b'], self.left.interfaces['crotch_b']),
        )

        self.interfaces = {
            'top_f': pyp.Interface.from_multiple(
                self.right.interfaces['top_f'], self.left.interfaces['top_f']),
            'top_b': pyp.Interface.from_multiple(
                self.right.interfaces['top_b'], self.left.interfaces['top_b']),
            # Some are reversed for correct connection
            'top': pyp.Interface.from_multiple(   # around the body starting from front right
                self.right.interfaces['top_f'],
                self.left.interfaces['top_f'].reverse(),
                self.left.interfaces['top_b'],   
                self.right.interfaces['top_b'].reverse()),
        }

class WBPants(pyp.Component):
    def __init__(self, body, design) -> None:
        super().__init__('WBPants')

        self.pants = Pants(body, design)

        # pants top
        wb_len = (self.pants.interfaces['top_b'].projecting_edges().length() + 
                    self.pants.interfaces['top_f'].projecting_edges().length())

        self.wb = bands.StraightWB(body, design)
        self.wb.translate_by([0, self.wb.width + 2, 0])

        self.stitching_rules = pyp.Stitches(
            (self.pants.interfaces['top'], self.wb.interfaces['bottom']),
        )


"""
    Module contains classes needed to simulate garments from patterns in Maya.
    Note that Maya uses Python 2.7 (incl Maya 2020) hence this module is adapted to Python 2.7
"""
# Basic
from __future__ import print_function
from __future__ import division
from functools import partial
import errno
import json
import numpy as np
import os
import time

# Maya
from maya import cmds
from maya import OpenMaya

# Arnold
import mtoa.utils as mutils
from mtoa.cmds.arnoldRender import arnoldRender
import mtoa.core as mtoa

# My modules
import pattern.core as core
from mayaqltools import qualothwrapper as qw
reload(core)
reload(qw)


class MayaGarment(core.ParametrizedPattern):
    """
    Extends a pattern specification in custom JSON format to work with Maya
        Input:
            * Pattern template in custom JSON format
        * import panel to Maya scene TODO
        * cleaning imported stuff TODO
        * Basic operations on panels in Maya TODO
    """
    def __init__(self, pattern_file, clean_on_die=False):
        super(MayaGarment, self).__init__(pattern_file)
        self.self_clean = clean_on_die

        self.last_verts = None
        self.current_verts = None
        self.loaded_to_maya = False
        self.obstacles = []
        self.shader = None
        self.MayaObjects = {}
        self.sim_material = {}
    
    def __del__(self):
        """Remove Maya objects when dying"""
        if self.self_clean:
            self.clean(True)

    def load(self, obstacles=[], shader=None, material={}, parent_group=None):
        """
            Loads current pattern to Maya as simulatable garment.
            If already loaded, cleans previous geometry & reloads
        """
        if self.loaded_to_maya:
            self.sim_material = self.fetchMaterialSimProps()  # save the latest material
        self.clean(True)
        
        self.load_panels(parent_group)
        self.stitch_panels()
        self.loaded_to_maya = True

        self.setShader(shader)
        self.add_colliders(obstacles)
        self.setMaterialSimProps(material)

        print('Garment ' + self.name + ' is loaded to Maya')

    def load_panels(self, parent_group=None):
        """Load panels to Maya as curve collection & geometry objects.
            Groups them by panel and by pattern"""
        # top group
        group_name = cmds.group(em=True, n=self.name)  # emrty at first
        if parent_group is not None:
            group_name = cmds.parent(group_name, parent_group)
        self.MayaObjects['pattern'] = group_name
        
        # Load panels as curves
        self.MayaObjects['panels'] = {}
        for panel_name in self.pattern['panels']:
            panel_maya = self._load_panel(panel_name, group_name)
    
    def setShader(self, shader=None):
        """
            Sets material properties for the cloth object created from current panel
        """
        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot set materials')

        # TODO accept input from file
        cloth = self.get_qlcloth_props_obj()

        # Controls stretchness of the fabric
        cmds.setAttr(cloth + '.stretch', 100)

        # Friction between cloth and itself 
        # (friction with body controlled by collider props)
        cmds.setAttr(cloth + '.friction', 0.25)

        if shader is not None:  # use previous othervise
            self.shader = shader

        if self.shader is not None:
            cmds.select(self.get_qlcloth_geomentry())
            cmds.hyperShade(assign=self.shader)

    def add_colliders(self, obstacles=[]):
        """
            Adds given Maya objects as colliders of the garment
        """
        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot load colliders')
        if obstacles:  # if not given, use previous ones
            self.obstacles = obstacles

        for obj in self.obstacles:
            collider = qw.qlCreateCollider(
                self.get_qlcloth_geomentry(), 
                obj
            )
            # properties
            # TODO experiment with the value -- it's now set arbitrarily
            qw.setColliderFriction(collider, 0.5)
            # organize object tree
            cmds.parent(collider, self.MayaObjects['pattern'])

    def clean(self, delete=False):
        """ Hides/removes the garment from Maya scene 
            NOTE all of the maya ids assosiated with the garment become invalidated, 
            if delete flag is True
        """
        if self.loaded_to_maya:
            # Remove from simulation
            cmds.setAttr(self.get_qlcloth_props_obj() + '.active', 0)
            if delete:
                print('MayaGarment::Deleting {}'.format(self.MayaObjects['pattern']))
                cmds.delete(self.MayaObjects['pattern'])
                qw.deleteSolver()

                self.loaded_to_maya = False
                self.MayaObjects = {}  # clean 
            else:
                cmds.hide(self.MayaObjects['pattern'])                

        # do nothing if not loaded -- already clean =)

    def fetchMaterialSimProps(self):
        """Return simulation properties"""
        return qw.fetchMaterialProps(self.get_qlcloth_props_obj())

    def setMaterialSimProps(self, props={}):
        """Pass material properties for cloth to Qualoth"""
        if props:
            self.sim_material = props
        qw.setMaterialProps(
            self.get_qlcloth_props_obj(), 
            self.sim_material
        )

    def get_qlcloth_geomentry(self):
        """
            Find the first Qualoth cloth geometry object belonging to current pattern
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')

        if 'qlClothShape' not in self.MayaObjects:
            children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
            cloths = [obj for obj in children 
                      if 'qlCloth' in obj and 'Out' in obj and 'Shape' in obj]
            self.MayaObjects['qlClothShape'] = cloths[0]

        return self.MayaObjects['qlClothShape']

    def get_qlcloth_props_obj(self):
        """
            Find the first qlCloth object belonging to current pattern
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')

        if 'qlCloth' not in self.MayaObjects:
            children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
            cloths = [obj for obj in children 
                      if 'qlCloth' in obj and 'Out' not in obj and 'Shape' in obj]
            self.MayaObjects['qlCloth'] = cloths[0]

        return self.MayaObjects['qlCloth']

    def get_qlcloth_geom_dag(self):
        """
            returns DAG reference to cloth shape object
        """
        if not self.loaded_to_maya:
            raise RuntimeError('MayaGarmentError::Pattern is not yet loaded.')

        if 'shapeDAG' not in self.MayaObjects:
            # https://help.autodesk.com/view/MAYAUL/2016/ENU/?guid=__files_Maya_Python_API_Using_the_Maya_Python_API_htm
            selectionList = OpenMaya.MSelectionList()
            selectionList.add(self.get_qlcloth_geomentry())
            self.MayaObjects['shapeDAG'] = OpenMaya.MDagPath()
            selectionList.getDagPath(0, self.MayaObjects['shapeDAG'])

        return self.MayaObjects['shapeDAG']

    def update_verts_info(self):
        """
            Retrieves current vertex positions from Maya & updates the last state.
            For best performance, should be called on each iteration of simulation
            Assumes the object is already loaded & stitched
        """
        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot update verts info')

        # working with meshes http://www.fevrierdorian.com/blog/post/2011/09/27/Quickly-retrieve-vertex-positions-of-a-Maya-mesh-%28English-Translation%29
        cloth_dag = self.get_qlcloth_geom_dag()
        
        mesh = OpenMaya.MFnMesh(cloth_dag)
        maya_vertices = OpenMaya.MPointArray()
        mesh.getPoints(maya_vertices, OpenMaya.MSpace.kWorld)

        vertices = np.empty((maya_vertices.length(), 3))
        for i in range(maya_vertices.length()):
            for j in range(3):
                vertices[i, j] = maya_vertices[i][j]

        self.last_verts = self.current_verts
        self.current_verts = vertices

    def is_static(self, threshold):
        """
            Checks wether garment is in the static equilibrium
            Compares current state with the last recorded state
        """
        if not self.loaded_to_maya:
            raise RuntimeError(
                'MayaGarmentError::Pattern is not yet loaded. Cannot check static')
        
        if self.last_verts is None:  # first iteration
            return False
        
        # Compare L1 norm per vertex
        # Checking vertices change is the same as checking if velocity is zero
        diff = np.abs(self.current_verts - self.last_verts)
        diff_L1 = np.sum(diff, axis=1)
        # DEBUG print(np.sum(diff), threshold * len(diff))
        if (diff_L1 < threshold).all():  # compare vertex-vize to be independent of #verts
            return True
        else:
            return False

    def is_penetrating(self, obstacles=[]):
        """Checks wheter garment intersects given obstacles or
        its colliders if obstacles are not given
        NOTE Implementation is lazy & might have false negatives
        TODO proper penetration check
        """
        raise NotImplementedError()

        if not obstacles:
            obstacles = self.obstacles
        
        print('Penetration check')

        for obj in obstacles:
            # experiment on copies
            obj_2 = cmds.duplicate(obj)[0]
            cloth_2 = cmds.duplicate(self.get_qlcloth_geomentry())[0]

            intersect = cmds.polyBoolOp(cloth_2, obj_2, op=3)
            print(intersect)

            # check if empty
            print(cmds.polyEvaluate(intersect[0], t=True))

            # delete all the extra objects
            # cmds.delete(obj_2)
            # cmds.delete(cloth_2)
            # cmds.delete(intersect)

    def sim_caching(self, caching=True):
        """Toggles the caching of simulation steps to garment folder"""
        if caching:
            # create folder
            self.cache_path = os.path.join(self.path, self.name + '_simcache')
            try:
                os.makedirs(self.cache_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:  # ok if directory exists
                    raise
                pass
        else:
            # disable caching
            self.cache_path = ''            

    def save_mesh(self, folder=''):
        """
            Saves cloth as obj file to a given folder or 
            to the folder with the pattern if not given.
        """
        if not self.loaded_to_maya:
            print('MayaGarmentWarning::Pattern is not yet loaded. Nothing saved')
            return

        if folder:
            filepath = folder
        else:
            filepath = self.path
        self._save_to_path(filepath, self.name + '_sim')

    def cache_if_enabled(self, frame):
        """If caching is enabled -> saves current geometry to cache folder
            Does nothing otherwise """
        if not self.loaded_to_maya:
            print('MayaGarmentWarning::Pattern is not yet loaded. Nothing cached')
            return

        if hasattr(self, 'cache_path') and self.cache_path:
            self._save_to_path(self.cache_path, self.name + '_{:04d}'.format(frame))

    def _load_panel(self, panel_name, pattern_group=None):
        """
            Loads curves contituting given panel to Maya. 
            Goups them per panel
        """
        panel = self.pattern['panels'][panel_name]
        vertices = np.asarray(panel['vertices'])
        self.MayaObjects['panels'][panel_name] = {}
        self.MayaObjects['panels'][panel_name]['edges'] = []

        # top panel group
        panel_group = cmds.group(n=panel_name, em=True)
        if pattern_group is not None:
            panel_group = cmds.parent(panel_group, pattern_group)[0]
        self.MayaObjects['panels'][panel_name]['group'] = panel_group

        # draw edges
        curve_names = []
        for edge in panel['edges']:
            curve_points = self._edge_as_3d_tuple_list(edge, vertices)
            curve = cmds.curve(p=curve_points, d=(len(curve_points) - 1))
            curve_names.append(curve)
            self.MayaObjects['panels'][panel_name]['edges'].append(curve)
        # Group  
        curve_group = cmds.group(curve_names, n=panel_name + '_curves')
        curve_group = cmds.parent(curve_group, panel_group)[0]
        self.MayaObjects['panels'][panel_name]['curve_group'] = curve_group
        # 3D placemement
        self._apply_panel_3d_placement(panel_name)

        # Create geometry
        panel_geom = qw.qlCreatePattern(curve_group)
        # take out the solver node -- created only once per scene, no need to store
        solvers = [obj for obj in panel_geom if 'Solver' in obj]
        panel_geom = list(set(panel_geom) - set(solvers))
        cmds.parent(panel_geom, panel_group)  # organize

        return panel_group

    def _edge_as_3d_tuple_list(self, edge, vertices):
        """
            Represents given edge object as list of control points
            suitable for draing in Maya
        """
        points = vertices[edge['endpoints'], :]
        if 'curvature' in edge:
            control_coords = self._control_to_abs_coord(
                points[0], points[1], edge['curvature']
            )
            # Rearrange
            points = np.r_[
                [points[0]], [control_coords], [points[1]]
            ]
        # to 3D
        points = np.c_[points, np.zeros(len(points))]

        return list(map(tuple, points))

    def _set_panel_3D_attr(self, panel_dict, panel_group, attribute, maya_attr):
        """Set recuested attribute to value from the spec"""
        if attribute in panel_dict:
            values = panel_dict[attribute]
        else:
            values = [0, 0, 0]
        cmds.setAttr(
            panel_group + '.' + maya_attr, 
            values[0], values[1], values[2],
            type='double3')

    def _apply_panel_3d_placement(self, panel_name):
        """Apply transform from spec to given panel"""
        panel = self.pattern['panels'][panel_name]
        panel_group = self.MayaObjects['panels'][panel_name]['curve_group']

        # set pivot to origin relative to currently loaded curves
        cmds.xform(panel_group, pivots=[0, 0, 0], worldSpace=True)

        # now place correctly
        self._set_panel_3D_attr(panel, panel_group, 'translation', 'translate')
        self._set_panel_3D_attr(panel, panel_group, 'rotation', 'rotate')

    def stitch_panels(self):
        """
            Create seams between qualoth panels.
            Assumes that panels are already loadeded (as curves).
            Assumes that after stitching every pattern becomes a single piece of geometry
            Returns
                Qulaoth cloth object name
        """
        self.MayaObjects['stitches'] = []
        for stitch in self.pattern['stitches']:
            from_curve = self._maya_curve_name(stitch['from'])
            # TODO add support for multiple "to" components
            to_curve = self._maya_curve_name(stitch['to'])
            stitch_id = qw.qlCreateSeam(from_curve, to_curve)
            stitch_id = cmds.parent(stitch_id, self.MayaObjects['pattern'])  # organization
            self.MayaObjects['stitches'].append(stitch_id[0])

        # after stitching, only one cloth\cloth shape object per pattern is left -- move up the hierarechy
        children = cmds.listRelatives(self.MayaObjects['pattern'], ad=True)
        cloths = [obj for obj in children if 'qlCloth' in obj]
        cmds.parent(cloths, self.MayaObjects['pattern'])

    def _maya_curve_name(self, address):
        """ Shortcut to retrieve the name of curve corresponding to the edge"""
        panel_name = address['panel']
        edge_id = address['edge']
        return self.MayaObjects['panels'][panel_name]['edges'][edge_id]

    def _save_to_path(self, path, filename):
        """Save current state of cloth object to given path with given filename as OBJ"""

        filepath = os.path.join(path, filename + '.obj')
        cmds.select(self.get_qlcloth_geomentry())
        cmds.file(
            filepath + '.obj',  # Maya 2020 stupidly cuts file extention 
            typ='OBJExport',
            es=1,  # export selected
            op='groups=0;ptgroups=0;materials=0;smoothing=0;normals=1',  # very simple obj
            f=1  # force override if file exists
        )


class MayaGarmentWithUI(MayaGarment):
    """Extension of MayaGarment that can generate GUI for controlling the pattern"""
    def __init__(self, pattern_file, clean_on_die=False):
        super(MayaGarmentWithUI, self).__init__(pattern_file, clean_on_die)
        self.ui_top_layout = None
        self.ui_controls = {}
        # TODO - move to Parametrized class
        self.edge_dirs_list = [
            'start', 
            'end', 
            'both'
        ]
    
    def __del__(self):
        super(MayaGarmentWithUI, self).__del__()
        # looks like UI now contains links to garment instance (callbacks, most probably)
        # If destructor is called, the UI is already clean
         
        # if self.ui_top_layout is not None:
        #     self._clean_layout(self.ui_top_layout)

    # ------- UI Drawing routines --------
    def drawUI(self, top_layout=None):
        """ Draw pattern controls in the given layout
            For correct connection with Maya attributes, it's recommended to call for drawing AFTER garment.load()
        """
        if top_layout is not None:
            self.ui_top_layout = top_layout
        if self.ui_top_layout is None:
            raise ValueError('GarmentDrawUI::top level layout not found')

        self._clean_layout(self.ui_top_layout)

        cmds.setParent(self.ui_top_layout)

        # Pattern name
        cmds.textFieldGrp(label='Pattern:', text=self.name, editable=False, 
                          cal=[1, 'left'], cw=[1, 50])

        # load panels info
        cmds.frameLayout(
            label='Panel Placement',
            collapsable=False, borderVisible=True,
            mh=10, mw=10
        )
        if not self.loaded_to_maya:
            cmds.text(label='<To be displayed after geometry load>')
        else:
            for panel in self.pattern['panels']:
                panel_layout = cmds.frameLayout(
                    label=panel, collapsable=True, collapse=True, borderVisible=True, mh=10, mw=10,
                    expandCommand=partial(cmds.select, self.MayaObjects['panels'][panel]['curve_group']),
                    collapseCommand=partial(cmds.select, self.MayaObjects['panels'][panel]['curve_group'])
                )
                self._ui_3d_placement(panel)
                cmds.setParent('..')
        cmds.setParent('..')

        # Parameters
        cmds.frameLayout(
            label='Parameters',
            collapsable=False, borderVisible=True,
            mh=10, mw=10
        )
        self._ui_params(self.parameters, self.spec['parameter_order'])
        cmds.setParent('..')

        # fin
        cmds.setParent('..')
        
    def _clean_layout(self, layout):
        """Removes all of the childer from layout"""
        # TODO make static or move outside? 
        children = cmds.layout(layout, query=True, childArray=True)
        if children:
            cmds.deleteUI(children)

    def _ui_3d_placement(self, panel_name):
        """Panel 3D placement"""
        if not self.loaded_to_maya:
            cmds.text(label='<To be displayed after geometry load>')

        # Position
        cmds.attrControlGrp(
            attribute=self.MayaObjects['panels'][panel_name]['curve_group'] + '.translate', 
            changeCommand=partial(self._panel_placement_callback, panel_name, 'translation', 'translate')
        )

        # Rotation
        cmds.attrControlGrp(
            attribute=self.MayaObjects['panels'][panel_name]['curve_group'] + '.rotate', 
            changeCommand=partial(self._panel_placement_callback, panel_name, 'rotation', 'rotate')
        )

    def _ui_param_value(self, param_name, param_range, value, idx=None, tag=''):
        """Create UI elements to display range and control the param value"""
        # range 
        cmds.rowLayout(numberOfColumns=3)
        cmds.text(label='Range ' + tag + ':')
        cmds.floatField(value=param_range[0], editable=False)
        cmds.floatField(value=param_range[1], editable=False)
        cmds.setParent('..')

        # value
        cmds.floatSliderGrp(
            label='Value ' + tag + ':', 
            field=True, value=value, 
            minValue=param_range[0], maxValue=param_range[1], 
            cal=[1, 'left'], cw=[1, 45], 
            changeCommand=partial(self._param_value_callback, param_name, idx),
            step=0.01
        )

    def _ui_params(self, params, order):
        """draw params UI"""
        # control
        cmds.button(label='To template state', 
                    backgroundColor=[227 / 256, 255 / 256, 119 / 256],
                    command=self._to_template_callback, 
                    ann='Snap all parameters to default values')
        cmds.button(label='Randomize', 
                    backgroundColor=[227 / 256, 186 / 256, 119 / 256],
                    command=self._param_randomization_callback, 
                    ann='Randomize all parameter values')

        # Parameters themselves
        for param_name in order:
            cmds.frameLayout(
                label=param_name, collapsable=True, collapse=True, mh=10, mw=10
            )
            # type 
            cmds.textFieldGrp(label='Type:', text=params[param_name]['type'], editable=False, 
                              cal=[1, 'left'], cw=[1, 30])

            # parameters might have multiple values
            values = params[param_name]['value']
            param_ranges = params[param_name]['range']
            if isinstance(values, list):
                ui_tags = ['X', 'Y', 'Z', 'W']
                for idx, (value, param_range) in enumerate(zip(values, param_ranges)):
                    self._ui_param_value(param_name, param_range, value, idx, ui_tags[idx])
            else:
                self._ui_param_value(param_name, param_ranges, values)

            # fin
            cmds.setParent('..')

    def _quick_dropdown(self, options, chosen='', label=''):
        """Add a dropdown with given options"""
        menu = cmds.optionMenu(label=label)
        for option in options:
            cmds.menuItem(label=option)
        if chosen:
            cmds.optionMenu(menu, e=True, value=chosen)

        return menu

    # -------- Callbacks -----------
    def _to_template_callback(self, *args):
        """Returns current pattern to template state and 
        updates UI accordingly"""
        # update
        print('Pattern returns to origins..')
        self._restore_template()
        # update geometry in lazy manner
        if self.loaded_to_maya:
            self.load()
        # update UI in lazy manner
        self.drawUI()

    def _param_randomization_callback(self, *args):
        """Randomize parameter values & update everything"""
        # restore template state before making any changes to parameters
        self._restore_template(params_to_default=False)

        # get values
        self._randomize_parameters()
        
        # reapply all parameters
        self._update_pattern_by_param_values()
        
        # update geometry in lazy manner
        if self.loaded_to_maya:
            self.load()
        # update UI in lazy manner
        self.drawUI()

    def _param_value_callback(self, param_name, value_idx, *args):
        """Update pattern with new value"""
        # restore template state -- params are interdependent
        # change cannot be applied independently by but should follow specified param order
        self._restore_template(params_to_default=False)

        # get value
        new_value = args[0]
        # save value. No need to check ranges -- correct by UI
        if isinstance(self.parameters[param_name]['value'], list):
            self.parameters[param_name]['value'][value_idx] = new_value
        else:
            self.parameters[param_name]['value'] = new_value
        
        # reapply all parameters
        self._update_pattern_by_param_values()
        
        # update geometry in lazy manner
        if self.loaded_to_maya:
            self.load()

        # update values in UI too (lazy)
        self.drawUI()

    def _panel_placement_callback(self, panel_name, attribute, maya_attr):
        """Update pattern spec with tranlation/rotation info from Maya"""
        # get values
        values = cmds.getAttr(self.MayaObjects['panels'][panel_name]['curve_group'] + '.' + maya_attr)
        values = values[0]  # only one attribute requested

        # set values
        self.pattern['panels'][panel_name][attribute] = list(values)


class Scene(object):
    """
        Decribes scene setup that includes:
            * body object
            * floor
            * light(s) & camera(s)
        Assumes 
            * body the scene revolved aroung faces z+ direction
    """
    def __init__(self, body_obj, props, scenes_path='', clean_on_die=False):
        """
            Set up scene for rendering using loaded body as a reference
        """
        self.self_clean = clean_on_die

        self.props = props
        self.config = props['config']
        self.stats = props['stats']
        # load body to be used as a translation reference
        self.body_filepath = body_obj
        self.body = cmds.file(body_obj, i=True, rnn=True)[0]
        self.body = cmds.rename(self.body, 'body#')

        # Put camera. 
        self._add_camera()

        # scene
        self._init_arnold()
        self.scene = {}
        if 'scene' in self.config:
            self._load_maya_scene(os.path.join(scenes_path, self.config['scene']))
        else:
            self._simple_scene_setup()

    def __del__(self):
        """Remove all objects related to current scene if requested on creation"""
        if self.self_clean:
            cmds.delete(self.body)
            cmds.delete(self.camera)
            for key in self.scene:
                cmds.delete(self.scene[key])
                # garment color migh become invalid

    def _init_arnold(self):
        """Endure Arnold objects are launched in Maya"""

        objects = cmds.ls('defaultArnoldDriver')
        if not objects:  # Arnold objects not found
            # https://arnoldsupport.com/2015/12/09/mtoa-creating-the-defaultarnold-nodes-in-scripting/
            print('Initialized Arnold')
            mtoa.createOptions()

    def floor(self):
        return self.scene['floor']
    
    def cloth_shader(self):
        return self.scene['cloth_shader']

    def render(self, save_to, name='scene'):
        """
            Makes a rendering of a current scene, and saves it to a given path
        """

        # TODO saving for datasets in subfolders & not
        # Set saving to file
        filename = os.path.join(save_to, name)
        
        # https://forums.autodesk.com/t5/maya-programming/rendering-with-arnold-in-a-python-script/td-p/7710875
        # NOTE that attribute names depend on the Maya version. These are for Maya2020
        cmds.setAttr("defaultArnoldDriver.aiTranslator", "png", type="string")
        cmds.setAttr("defaultArnoldDriver.prefix", filename, type="string")

        start_time = time.time()
        im_size = self.config['resolution']

        arnoldRender(im_size[0], im_size[1], True, True, self.camera, ' -layer defaultRenderLayer')
        
        self.stats['render_time'].append(time.time() - start_time)

    def fetch_props_from_Maya(self):
        """Get properties records from Maya
            Note: it updates global config!"""
        self.config['body_color'] = self._fetch_color(self.scene['body_shader'])
        self.config['cloth_color'] = self._fetch_color(self.scene['cloth_shader'])
        self.config['floor_color'] = self._fetch_color(self.scene['floor_shader'])

        self.config['camera_rotation'] = cmds.getAttr(self.camera + '.rotate')[0]

    def _fetch_color(self, shader):
        """Return current color of a given shader node"""
        return cmds.getAttr(shader + '.color')[0]

    def _add_camera(self):
        """Puts camera in the scene
        NOTE Assumes body is facing +z direction"""

        aspect_ratio = self.config['resolution'][0] / self.config['resolution'][1]
        self.camera = cmds.camera(ar=aspect_ratio)
        self.camera = self.camera[0]

        cmds.setAttr(
            self.camera + '.rotate', 
            self.config['camera_rotation'][0], 
            self.config['camera_rotation'][1], 
            self.config['camera_rotation'][2], 
            type='double3')

        cmds.viewFit(self.camera, self.body, f=1)

    def _simple_scene_setup(self):
        """setup very simple scene & materials using info from props"""
        self.scene = {
            'floor': self._add_floor(self.body)
        }
        self.scene.update({
            # materials
            'body_shader': self._new_lambert(self.config['body_color'], self.body),
            'cloth_shader': self._new_lambert(self.config['cloth_color']),
            'floor_shader': self._new_lambert(self.config['floor_color'], self.scene['floor']),
            # light
            'light': mutils.createLocator('aiSkyDomeLight', asLight=True)
        })

    def _load_maya_scene(self, scenefile):
        """Load scene from external file. 
            NOTE Assumes certain naming of nodes in the scene!"""

        cmds.file(scenefile, i=True)
        self.scene = {
            'scene_group': cmds.ls('*scene*', transforms=True)[0],
            'floor': cmds.ls('*backdrop*', geometry=True)[0],
            'floor_shader': cmds.ls('*backdrop*', materials=True)[0],
            'body_shader': cmds.ls('*body*', materials=True)[0],
            'cloth_shader': cmds.ls('*garment*', materials=True)[0],
            'side_shader': cmds.ls('*wall*', materials=True)[0]
        }
        # apply coloring to body object
        if self.body:
            cmds.select(self.body)
            cmds.hyperShade(assign=self.scene['body_shader'])

        # adjust scene position s.t. body is standing in the middle
        body_low_center = self._get_object_lower_center(self.body)
        floor_low_center = self._get_object_lower_center(self.scene['floor'])
        old_translation = cmds.getAttr(self.scene['scene_group'] + '.translate')[0]
        
        cmds.setAttr(
            self.scene['scene_group'] + '.translate',
            old_translation[0] + body_low_center[0] - floor_low_center[0],
            old_translation[1] + body_low_center[1] - floor_low_center[1], 
            old_translation[2] + body_low_center[2] - floor_low_center[2],
            type='double3')  # apply to whole group s.t. lights positions were adjusted too

    def _get_object_lower_center(self, object):
        """return 3D position of the center of the lower side of bounding box"""
        bb = cmds.exactWorldBoundingBox(object)
        return [
            (bb[3] + bb[0]) / 2,
            bb[1],
            (bb[5] + bb[2]) / 2
        ]
    
    def _add_floor(self, target):
        """
            adds a floor under a given object
        """
        target_bb = cmds.exactWorldBoundingBox(target)

        size = 10 * (target_bb[4] - target_bb[1])
        floor = cmds.polyPlane(n='floor', w=size, h=size)

        # place under the body
        floor_level = target_bb[1]
        cmds.move((target_bb[3] + target_bb[0]) / 2,  # bbox center
                  floor_level, 
                  (target_bb[5] + target_bb[2]) / 2,  # bbox center
                  floor, a=1)

        return floor[0]

    def _new_lambert(self, color, target=None):
        """created a new shader node with given color"""
        shader = cmds.shadingNode('lambert', asShader=True)
        cmds.setAttr((shader + '.color'), 
                     color[0], color[1], color[2],
                     type='double3')

        if target is not None:
            cmds.select(target)
            cmds.hyperShade(assign=shader)

        return shader

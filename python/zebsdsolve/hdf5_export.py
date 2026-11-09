"""HDF5 containers for native export snapshots. No crystallographic calculations here."""
from pathlib import Path
import json
import os
import tempfile
import posixpath
import numpy as np

TSL = [0, 1, 2, 22, 4, 42, 3, 32, 6, 62, 23, 43]

def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HDF5 export requires h5py: pip install 'zebsdsolve[hdf5]'") from e
    return h5py

def _temporary(path):
    p = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + p.name + '.', suffix='.h5', dir=p.parent)
    os.close(fd)
    return name

def _string(group, name, value):
    return group.create_dataset(name, data=str(value), dtype=_h5py().string_dtype('utf-8'))

def _provenance(f, s, **extra):
    g = f.create_group('zebsdsolve')
    _string(g, 'Producer', 'zebsdsolve')
    _string(g, 'Version', s['version'])
    _string(g, 'SettingsJSON', json.dumps(s['settings']))
    _string(g, 'GeometryJSON', json.dumps(s['geometry']))
    _string(g, 'PhasesJSON', json.dumps(s['phases']))
    _string(g, 'CandidateScope', 'Best stored candidate per phase; no within-phase secondary peaks')
    _string(g, 'ProvenanceJSON', json.dumps({
        'schema_version': 1, 'version': s['version'], 'source_format': s['scan']['format'],
        'units': {'euler': 'radians', 'map': 'micrometres', 'lattice': 'angstrom, degrees'},
        **{key: s[key] for key in ('oxford_frame', 'calibration') if key in s},
    }))
    n = s['scan']['rows']*s['scan']['cols']
    selected = g.create_group('Selected')
    for key, data in s['results'].items():
        width = 4 if key == 'quaternions' else 3 if key == 'euler' else 1
        selected.create_dataset(key, data=np.asarray(data).reshape((n, width) if width > 1 else (n,)))
    c = s['candidates']
    if c['phase_count']:
        cg = g.create_group('Candidates')
        cg.create_dataset('PhaseCount', data=c['phase_count'])
        for key, data in c.items():
            if key == 'phase_count': continue
            width = 4 if key == 'quaternions' else 3 if key == 'euler' else 1
            shape = (c['phase_count'],n,width) if width>1 else (c['phase_count'],n)
            cg.create_dataset(key, data=np.asarray(data).reshape(shape))
    for key, value in extra.items():
        _string(g, key, json.dumps(value) if not isinstance(value, str) else value)
    return g

def _maps(s):
    r = {k: np.asarray(v) for k, v in s['results'].items()}
    n = len(r['phase'])
    r['euler'] = r['euler'].reshape(n, 3)
    r['quaternions'] = r['quaternions'].reshape(n, 4)
    yield r
    c = s['candidates']
    count = c['phase_count']
    if count < 2:
        return
    values = np.asarray(c[s['settings']['phase_by']], dtype=float).reshape(count, n)
    valid = np.asarray(c['sweep']).reshape(count, n) >= 0
    valid &= np.arange(count)[:, None] != r['phase'][None, :]
    order = np.argsort(-np.where(valid, values, -np.inf), axis=0, kind='stable')
    arrays = {k: np.asarray(v).reshape((count, n, 3 if k == 'euler' else 4) if k in ('euler', 'quaternions') else (count, n))
              for k, v in c.items() if k != 'phase_count'}
    for rank in range(count - 1):
        phase = order[rank]
        exists = valid[phase, np.arange(n)]
        out = {k: v[phase, np.arange(n)].copy() for k, v in arrays.items()}
        out['phase'] = np.where(exists, phase, -1)
        for k in out:
            if k != 'phase':
                out[k][~exists] = -1 if k == 'sweep' else 0
        yield out

def write_emsphinx(path, s):
    h5py = _h5py()
    if len(s['phases']) > 255:
        raise ValueError('EMSphInx phase IDs support at most 255 phases')
    rows, cols = s['scan']['rows'], s['scan']['cols']
    n = rows * cols
    temporary = _temporary(path)
    try:
        with h5py.File(temporary, 'w') as f:
            _string(f, 'Manufacturer', 'EMSphInx')  # Native reader's format discriminator.
            f.attrs['Producer'] = 'zebsdsolve'
            _provenance(f, s, Frame='Same raw Bunge/raster conventions as ANG; MTEX setting 2',
                        Ranking='Scan 1 selected; later scans alternate phases ranked by ' + s['settings']['phase_by'])
            for rank, r in enumerate(_maps(s), 1):
                ebsd = f.create_group(f'Scan {rank}/EBSD')
                header = ebsd.create_group('Header')
                for key, value in [('nRows', rows), ('nColumns', cols), ('Step X', s['geometry']['step_x_um']), ('Step Y', s['geometry']['step_y_um'])]:
                    header.create_dataset(key, data=value)
                _string(header, 'Grid Type', 'SqrGrid')
                _string(header, 'Scan ID', f'Match {rank}')
                header.create_dataset('Sample Tilt', data=s['geometry']['sample_tilt_deg'])
                pc = s['geometry']['pc']
                ratio = s['scan']['pat_w'] / s['scan']['pat_h']
                calibration = header.create_group('Pattern Center Calibration')
                for key, value in zip(['x-star', 'y-star', 'z-star'], [pc[0], (1-pc[1])/ratio, pc[2]/ratio]):
                    calibration.create_dataset(key, data=value)
                phases = header.create_group('Phase')
                for i, phase in enumerate(s['phases'], 1):
                    group = phases.create_group(str(i))
                    _string(group, 'MaterialName', phase['name'])
                    group.create_dataset('Symmetry', data=TSL[phase['laue']])
                    group.create_dataset('Space Group', data=phase['space_group'])
                    for key, v in zip(['a', 'b', 'c', 'alpha', 'beta', 'gamma'], phase['lattice']):
                        group.create_dataset('Lattice Constant '+key, data=v)
                data = ebsd.create_group('Data')
                data.create_dataset('Phase', data=np.where(r['phase'] < 0, 255, r['phase']).astype('u1'))
                for k, key in enumerate(['Phi1', 'Phi', 'Phi2']):
                    data.create_dataset(key, data=r['euler'][:, k].astype('f8'))
                for key, source in [('CI', 'ncc'), ('Metric', 'ncc'), ('IQ', 'nmi'), ('NMI', 'nmi'), ('Z', 'z'), ('Sweep', 'sweep')]:
                    data.create_dataset(key, data=r[source])
                data.create_dataset('PixelID', data=np.arange(n, dtype='u8'))
                data.create_dataset('Quaternion', data=r['quaternions'])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def _scalar(group, name):
    return np.asarray(group[name][()]).reshape(-1)[0]

def _unit(dataset):
    unit = dataset.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    return str(unit).strip().lower()

def _lattice(group):
    d = group['Lattice Dimensions']
    unit = d.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    factor = {'angstrom': 1, 'angstroms': 1, 'å': 1, 'nm': 10}.get(str(unit).lower())
    if factor is None:
        raise ValueError('Unknown Oxford lattice length unit')
    angles = group['Lattice Angles']
    unit = angles.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    if str(unit).lower() not in ('rad', 'radian', 'radians'):
        raise ValueError('Oxford lattice angles must be radians')
    return np.r_[np.asarray(d[()]).reshape(3)*factor, np.rad2deg(np.asarray(angles[()]).reshape(3))]

def _matches(group, phase, allow_laue=False):
    sg = int(_scalar(group, 'Space Group')) if 'Space Group' in group else 0
    laue = int(_scalar(group, 'Laue Group')) if 'Laue Group' in group else 0
    return (sg > 0 and sg == phase['space_group'] or (allow_laue and sg == 0 and laue == phase['laue'])) and np.allclose(_lattice(group), phase['lattice'], rtol=1e-5, atol=1e-4)

PATTERN_DATASETS = {'Processed Patterns', 'Unprocessed Patterns', 'Raw Patterns'}

def _copy_without_patterns(source, target, relocations=None):
    """Native HDF5 copies retain metadata; diffraction payloads are never read."""
    h5py = _h5py()
    relocations = relocations or {}
    def mapped(path):
        path = '/'+ '/'.join(p for p in path.split('/') if p and p != '.')
        for old, new in relocations.items():
            if path == old or path.startswith(old+'/'):
                return new+path[len(old):]
        return path
    def identity(obj):
        return h5py.h5o.get_info(obj.id).addr
    def references(dtype):
        if h5py.check_dtype(ref=dtype) is not None: return True
        if dtype.fields: return any(references(v[0]) for v in dtype.fields.values())
        if dtype.subdtype: return references(dtype.subdtype[0])
        base = h5py.check_dtype(vlen=dtype)
        if isinstance(base, np.dtype): return references(base)
        return False
    omitted_ids, omitted, checked = set(), [], set()
    def preflight(g):
        key = identity(g)
        if key in checked: return
        checked.add(key)
        for name in g.attrs:
            if references(g.attrs.get_id(name).dtype):
                raise ValueError('Oxford metadata export does not support HDF5 reference attributes: '+g.name)
        if isinstance(g, h5py.Dataset):
            if references(g.dtype):
                raise ValueError('Oxford metadata export does not support HDF5 reference datasets: '+g.name)
            return
        for name in g:
            link = g.get(name, getlink=True)
            if isinstance(link,h5py.SoftLink) and '..' in link.path.split('/'):
                raise ValueError('Oxford metadata export does not support literal .. in HDF5 soft links: '+g.name+'/'+name)
            if name in PATTERN_DATASETS:
                if isinstance(link, h5py.ExternalLink):
                    omitted.append(posixpath.join(g.name,name)); continue
                obj = local_target(posixpath.join(g.name,name))
                if isinstance(obj,h5py.Dataset):
                    omitted_ids.add(identity(obj)); omitted.append(posixpath.join(g.name,name)); continue
            if not isinstance(link, h5py.HardLink): continue
            obj = g[name]
            preflight(obj)
    def local_target(path, seen=None):
        # Resolve local links only; external files are never opened.
        seen = set() if seen is None else seen
        path = '/'+ '/'.join(p for p in path.split('/') if p and p != '.')
        if path in seen: return None
        seen.add(path)
        parts = path.strip('/').split('/') if path != '/' else []
        obj = source
        for i, name in enumerate(parts):
            if not isinstance(obj, h5py.Group): return None
            link = obj.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink) or link is None: return None
            if isinstance(link, h5py.SoftLink):
                target_path = link.path if link.path.startswith('/') else posixpath.join(obj.name, link.path)
                return local_target(posixpath.join(target_path, *parts[i+1:]), seen)
            obj = obj[name]
        return obj
    preflight(source)
    relocated_ids = {}
    def relocated_objects(obj,path):
        key = identity(obj)
        if key in relocated_ids: return
        relocated_ids[key] = mapped(path)
        if isinstance(obj,h5py.Group):
            for name in obj:
                if isinstance(obj.get(name,getlink=True),h5py.HardLink):
                    relocated_objects(obj[name],posixpath.join(path,name))
    for old in relocations:
        obj = local_target(old)
        if obj is not None: relocated_objects(obj,old)
    copied = {identity(source): target.name}
    def copy_attrs(a,b):
        for name in a.attrs:
            attr = a.attrs.get_id(name)
            b.attrs.create(name, a.attrs[name], shape=attr.shape, dtype=attr.dtype)
    def copy_group(a,b):
        copy_attrs(a,b)
        for name in a:
            link = a.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                if name in PATTERN_DATASETS: continue
                b[name] = link
                continue
            if isinstance(link, h5py.SoftLink):
                path = link.path if link.path.startswith('/') else posixpath.join(a.name,link.path)
                obj = local_target(path)
                if obj is not None and identity(obj) in omitted_ids:
                    omitted.append(posixpath.join(a.name,name)); continue
                # Archived Data Processing links must retain their original targets
                # after the group moves, including relative links out of that group.
                if obj is not None and identity(obj) in relocated_ids:
                    b[name] = h5py.SoftLink(relocated_ids[identity(obj)])
                elif mapped(a.name) != a.name or mapped(path) != path:
                    b[name] = h5py.SoftLink(mapped(path))
                else:
                    b[name] = link
                continue
            obj = a[name]; key = identity(obj)
            if key in omitted_ids:
                omitted.append(obj.name); continue
            if key in copied:
                b[name] = target[copied[key]]
            elif isinstance(obj, h5py.Group):
                group = b.create_group(name)
                copied[key] = group.name
                copy_group(obj,group)
            else:
                source.copy(obj,b,name=name)
                copied[key] = b[name].name
    copy_group(source,target)
    return sorted(set(omitted))

def _oxford_group(f, s):
    found = []
    f.visititems(lambda name, obj: found.append(name) if isinstance(obj, _h5py().Group) and name.endswith('/EBSD') and 'Data/Euler' in obj else None)
    if len(found) != 1:
        raise ValueError('Oxford export requires one unambiguous EBSD acquisition')
    g = f[found[0]]
    header, data = g['Header'], g['Data']
    n = s['scan']['rows']*s['scan']['cols']
    if (int(_scalar(header, 'X Cells')), int(_scalar(header, 'Y Cells'))) != (s['scan']['cols'], s['scan']['rows']):
        raise ValueError('Oxford baseline map dimensions differ from indexed scan')
    if data['Euler'].shape != (n, 3) or data['Phase'].shape != (n,):
        raise ValueError('Oxford baseline result layout differs from indexed scan')
    if data['Euler'].dtype.kind != 'f' or data['Phase'].dtype.kind not in 'ui':
        raise ValueError('Unsupported Oxford orientation/phase datatype')
    for key, expected in [('Pattern Width',s['scan']['pat_w']),('Pattern Height',s['scan']['pat_h'])]:
        if key in header and int(_scalar(header,key)) != expected:
            raise ValueError('Oxford baseline detector dimensions differ from indexed patterns')
    euler_unit = data['Euler'].attrs.get('Unit', '')
    if isinstance(euler_unit, bytes): euler_unit = euler_unit.decode()
    if str(euler_unit).lower() not in ('rad', 'radian', 'radians'):
        raise ValueError('Oxford baseline Euler unit must be radians')
    for axis, expected in [('X', np.tile(np.arange(s['scan']['cols']), s['scan']['rows'])), ('Y', np.repeat(np.arange(s['scan']['rows']), s['scan']['cols']))]:
        if axis not in data:
            raise ValueError('Oxford baseline requires per-pixel '+axis+' for a verified pixel join')
        microns = ('um','µm','μm','micrometres','micrometers')
        if _unit(data[axis]) not in microns or _unit(header[axis+' Step']) not in microns:
            raise ValueError('Oxford positions and raster steps must declare micrometre units')
        step = float(_scalar(header, axis+' Step'))
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Invalid Oxford raster step')
        if not np.isclose(step, s['geometry']['step_'+axis.lower()+'_um'], rtol=1e-5, atol=1e-7):
            raise ValueError('Oxford baseline and indexed map steps differ')
        positions = np.asarray(data[axis][()]).reshape(n)
        if not np.allclose(positions, positions[0]+expected*step, rtol=0, atol=max(1e-6, abs(step)*1e-3)):
            raise ValueError('Oxford positions do not match engine row-major pixel IDs')
    return g.name

def write_oxford(path, reference, s, engine_to_cs1, frame_provenance, phase_id_map=None):
    h5py = _h5py()
    q = np.asarray(engine_to_cs1, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(float(q@q)-1)>1e-6 or not str(frame_provenance).strip():
        raise ValueError('OxfordFrameUnresolved: native unit engine-to-CS1 rotation and provenance required')
    if Path(path).resolve() == Path(reference).resolve() or (Path(path).exists() and os.path.samefile(path, reference)):
        raise ValueError('Oxford export must not overwrite its original baseline')
    with h5py.File(reference, 'r') as f:
        if 'zebsdsolve' in f:
            raise ValueError('Use the original acquisition as baseline, not a prior zebsdsolve export')
        name = _oxford_group(f, s)
        phases = f[name+'/Header/Phases']
        ids = sorted(int(k) for k in phases.keys())
        mapping, additions = [], []
        next_id = max(ids, default=0)+1
        if phase_id_map is not None and len(phase_id_map) != len(s['phases']):
            raise ValueError('phase_id_map length differs from indexed phases')
        for i, phase in enumerate(s['phases']):
            matches = [pid for pid in ids if _matches(phases[str(pid)], phase)]
            if phase_id_map is not None:
                pid = int(phase_id_map[i])
                if pid not in ids or not _matches(phases[str(pid)], phase, allow_laue=True): raise ValueError('Explicit Oxford phase mapping conflicts with crystal metadata')
            elif len(matches) == 1:
                pid = matches[0]
            elif len(matches) > 1:
                raise ValueError('Ambiguous Oxford phases; supply phase_id_map')
            else:
                pid = next_id; next_id += 1; additions.append((pid, phase))
            mapping.append(pid)
        if len(set(mapping)) != len(mapping):
            raise ValueError('Each indexed phase must map to a distinct Oxford phase')
        dtype = f[name+'/Data/Phase'].dtype
        if dtype.kind not in 'ui' or max(mapping, default=0) > np.iinfo(dtype).max:
            raise ValueError('New phase IDs exceed the original Oxford Phase datatype')
    temporary = _temporary(path)
    try:
        with h5py.File(reference, 'r') as source, h5py.File(temporary, 'w') as output:
            processing = name.rsplit('/',1)[0]+'/Data Processing'
            relocations = {processing:'/zebsdsolve/OriginalDataProcessing'} if processing in source else {}
            omitted_patterns = _copy_without_patterns(source, output, relocations)
        with h5py.File(temporary, 'r+') as f:
            root = _provenance(f, s, Frame='Oxford CS1', EngineToCS1=q.tolist(), FrameProvenance=frame_provenance, PhaseIDMap=mapping,
                               PatternData='Omitted; metadata and map results only', OmittedPatternPaths=omitted_patterns)
            old_processing = name.rsplit('/', 1)[0]+'/Data Processing'
            if old_processing in f:
                f.move(old_processing, root.name+'/OriginalDataProcessing')
            calibration = s.get('calibration', {})
            header = f[name+'/Header']
            for key, field, count in [('Detector Orientation Euler', 'effective_detector_euler_radians', 3),
                                      ('Tilt Angle', 'effective_tilt_radians', 1)]:
                if field not in calibration:
                    continue
                values = np.asarray(calibration[field], dtype=float).reshape(-1)
                if values.size != count or not np.isfinite(values).all():
                    raise ValueError('Invalid native effective Oxford calibration: '+field)
                if key not in header:
                    raise ValueError('Oxford source calibration header is missing: '+key)
                dataset = header[key]
                if dataset.size != count or dataset.dtype.kind != 'f' or _unit(dataset) not in ('rad', 'radians'):
                    raise ValueError('Unsupported Oxford calibration layout or units: '+key)
                original = root.require_group('OriginalCalibration')
                f.copy(dataset, original, name=key)
                dataset[...] = values.reshape(dataset.shape)
            data = f[name+'/Data']; phases = f[name+'/Header/Phases']
            for pid, phase in additions:
                g = phases.create_group(str(pid))
                _string(g, 'Phase Name', phase['name'])
                _string(g, 'Reference', 'zebsdsolve master pattern: '+phase['name'])
                g.create_dataset('Space Group', data=[phase['space_group']])
                g.create_dataset('Laue Group', data=[phase['laue']])
                g.create_dataset('Lattice Dimensions', data=[phase['lattice'][:3]]).attrs['Unit'] = 'angstrom'
                g.create_dataset('Lattice Angles', data=[np.deg2rad(phase['lattice'][3:])]).attrs['Unit'] = 'rad'
            old = root.create_group('OriginalIndexing')
            r = s['results']; valid = np.asarray(r['phase']) >= 0
            root.create_dataset('UpdatedPixelMask', data=valid.astype('u1'))
            # Keep source datasets/attributes exactly; only indexed rows change in place.
            for key in ['Euler', 'Phase', 'Mean Angular Deviation', 'Bands', 'Error']:
                if key in data:
                    f.copy(data[key], old, name=key)
                    values = data[key][()]
                    if key == 'Euler': values[valid] = np.asarray(r['euler']).reshape(-1, 3)[valid]
                    elif key == 'Phase': values[valid] = np.asarray(mapping)[np.asarray(r['phase'])[valid]]
                    else: values[valid] = np.nan if values.dtype.kind == 'f' else 0
                    data[key][...] = values
            pc = np.asarray(s['pattern_centres']).reshape(-1, 3)
            ratio = s['scan']['pat_h']/s['scan']['pat_w']
            for key, values in [('Pattern Center X', pc[:, 0]), ('Pattern Center Y', (1-pc[:, 1])*ratio), ('Detector Distance', pc[:, 2]*ratio)]:
                if key in data:
                    f.copy(data[key], old, name=key)
                    original = data[key][()]
                    if original.shape != valid.shape:
                        raise ValueError('Oxford calibration layout differs from indexed pixels')
                    original[valid] = values[valid]
                    data[key][...] = original
                else:
                    out = np.full(len(valid), np.nan)
                    out[valid] = values[valid]
                    data.create_dataset(key, data=out).attrs['Unit'] = 'fraction'
            for key, source in [('NCC', 'ncc'), ('NMI', 'nmi'), ('Z', 'z')]:
                root.create_dataset(key, data=r[source])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)

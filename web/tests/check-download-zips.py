"""Validate saved browser ZIP artifacts with zipfile, h5py, NumPy and Pillow.

Usage: python web/tests/check-download-zips.py /path/printed/by/batch-downloads-browser.mjs
Checks archive structure, serialized records, maps and project state.
Independent scientific importer validation is private development work.
"""
import io,json,sys,zipfile
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
folder=Path(sys.argv[1]);archives=sorted(folder.glob('*.zip'))
assert archives,'No browser downloads found'
for path in archives:
    stem=path.stem
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None,path
        entries=archive.namelist()
        vendor='.h5oina' if any(name.endswith('.h5oina') for name in entries) else '.ang'
        assert sorted(entries)==sorted(f'{stem}/{stem}{suffix}' for suffix in ('.h5',vendor,'_IPF-Z.png','_NCC.png','.json')),(path,entries)
        assert all(info.compress_type==zipfile.ZIP_STORED for info in archive.infolist())
        project=json.loads(archive.read(f'{stem}/{stem}.json'))
        assert project['batch']['transport']=='downloads' and project['batch']['status']=='archive-ready'
        assert len(project['batch']['outputs'])==5 and all(entry['complete'] for entry in project['batch']['outputs'])
        rows,cols=project['scan_info']['rows'],project['scan_info']['cols']
        with h5py.File(io.BytesIO(archive.read(f'{stem}/{stem}.h5'))) as result:
            assert result['zebsdsolve/Maps/IPFZ_RGBA'].shape==(rows,cols,4)
            assert result['zebsdsolve/Maps/NCC'].shape==(rows,cols)
            for suffix in ('_IPF-Z.png','_NCC.png'):
                image=Image.open(io.BytesIO(archive.read(f'{stem}/{stem}{suffix}'))).convert('RGBA')
                assert image.size==(cols,rows)
                key='IPFZ_RGBA' if suffix=='_IPF-Z.png' else 'NCC_RGBA'
                np.testing.assert_array_equal(np.asarray(image),result['zebsdsolve/Maps/'+key][()])
        vendor_bytes=archive.read(f'{stem}/{stem}{vendor}')
        if vendor=='.ang':
            records=np.loadtxt(io.BytesIO(vendor_bytes),ndmin=2)
            assert records.shape[0]==rows*cols,(path,records.shape,rows*cols)
            assert records.shape[1]>=10,(path,records.shape)
            assert np.isfinite(records[:,:5]).all(),path
        if vendor=='.h5oina':
            reference=json.loads((folder/'p6-reference.json').read_text())
            with h5py.File(io.BytesIO(vendor_bytes)) as result:
                root=next(group for group in result.values() if isinstance(group,h5py.Group) and 'EBSD' in group)
                data=root['EBSD/Data']
                assert data['Euler'].shape==(rows*cols,3)
                assert data['Phase'].shape==(rows*cols,)
                assert 'Processed Patterns' not in data and 'Unprocessed Patterns' not in data
                np.testing.assert_allclose(data['Euler'][reference['cell']],reference['euler'],atol=1e-6,rtol=0)
    print(path.name,': CRC, 5 matching members, native maps, JSON state and serialized vendor records passed')
print('All browser ZIP exports passed Python zipfile/h5py serialization validation.')

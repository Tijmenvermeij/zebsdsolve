import assert from 'node:assert/strict';
import { SourceFolders, verifyScanFolder, rememberFileHandle, filesFromHandles, fileHandleFor } from '../app/js/source-folders.js';
const missing=()=>Object.assign(new Error('missing'),{name:'NotFoundError'});
const makeFile=(name='scan.up1',time=123)=>new File(['1234567890'],name,{lastModified:time});
function handle(file,id){return {kind:'file',id,async getFile(){return file;},async isSameEntry(other){return id===other.id;}};}
function directory(name,id,entries){return {kind:'directory',name,id,entries,reads:0,async isSameEntry(other){return id===other.id;},async getFileHandle(name,options={}){assert.equal(options.create,undefined);this.reads++;if(!entries.has(name))throw missing();return entries.get(name);}};}
{
 const source=makeFile(), original=handle(source,'A/scan');rememberFileHandle(source,original);
 const a=directory('same-name','A',new Map([[source.name,original]]));
 const b=directory('same-name','B',new Map([[source.name,handle(makeFile(),'B/scan')]]));
 await assert.rejects(verifyScanFolder(b,source),/does not contain/);
 const cache=new SourceFolders();await assert.rejects(cache.accept(b,source,true),/does not contain/);assert.equal(cache.roots.length,0);
 assert.equal(await cache.accept(a,source,true),a);assert.equal(await cache.find(source),a);
 const alias=directory('same-name','A',a.entries);assert.equal(await cache.add(alias),a);assert.equal(cache.roots.length,1);
 const next=makeFile('second.up2'), nextHandle=handle(next,'A/next');a.entries.set(next.name,nextHandle);rememberFileHandle(next,nextHandle);
 assert.equal(await cache.find(next),a);
}
{
 const input=makeFile();const root=directory('scan-data','input',new Map([[input.name,handle(input,'input/file')]]));
 const cache=new SourceFolders();await cache.accept(root,input,true);
 assert.equal(await cache.find(input),root);
 assert.equal(await cache.find(makeFile()),null,'Matching metadata cannot route a different input-only File automatically');
 root.entries.set(input.name,handle(makeFile(input.name,124),'input/file'));
 await assert.rejects(cache.find(input),/differs/);
}
{
 const input=makeFile();const custom=directory('exports','custom',new Map());const cache=new SourceFolders();
 await cache.accept(custom,input,false);assert.equal(await cache.find(input),null);
 await assert.rejects(verifyScanFolder(custom,input),/containing scan.up1/);
}
{
 const file=makeFile(), entry=handle(file,'picked');const [picked]=await filesFromHandles([entry]);
 assert.equal(fileHandleFor(picked),entry);
}
console.log('Source folders: native identity, parent verification, alias handles, exact input fallback, changed sources and custom destinations passed.');

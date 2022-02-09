$MYPATH = "MYPATH"
$files = Get-ChildItem -PATH $MYPATH -FILE -Recurse -Include *.mp4, *.mkv, *.avi, *.wmv, *.ts, *.vob, *.divx, *.m4v, *.mpeg, *.rmvb, *.flv
$counter = 1
$totalNum = $files.Count
foreach ($file in $files)
{
    $filenameBase = $file.basename
    $filenameFull = $file.fullname.replace("\","/")
    $filedir = $file.DirectoryName
    echo "($counter/$totalNum) $filenameFull" 
    python3 bifgen.py -o "${filedir}\${filenameBase}-320-10.bif" "${filenameFull}"
    $counter++
}
$MYPATH = "MYPATH"
$files = Get-ChildItem -PATH $MYPATH -FILE -Recurse -Include *.mp4, *.mkv, *.avi, *.wmv, *.ts, *.vob, *.divx, *.m4v, *.mpeg, *.rmvb, *.flv
$counter = 1
$totalNum = $files.Count
foreach ($file in $files)
{
    $filenameBase = $file.basename
    $filenameFull = $file.fullname.replace("\","/")
    $filedir = $file.DirectoryName
    $BIF_PATH = "${filedir}\${filenameBase}-320-10.bif"
    echo "($counter/$totalNum) $filenameFull"
    if(Test-Path -Path $BIF_PATH -PathType Leaf){
        echo "BIF file does exist, skip !"
    }
    else {
        python3 bifgen.py -o $BIF_PATH "${filenameFull}"
    }
    $counter++
}
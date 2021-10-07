:: Create File List
echo file 0.mp4 >  mylist.txt 
echo file 1.mp4 >> mylist.txt
echo file 2.mp4 >> mylist.txt
echo file 3.mp4 >> mylist.txt
echo file 4.mp4 >> mylist.txt
echo file 5.mp4 >> mylist.txt
echo file 6.mp4 >> mylist.txt
echo file 7.mp4 >> mylist.txt
echo file 8.mp4 >> mylist.txt
echo file 9.mp4 >> mylist.txt

:: Concatenate Files
ffmpeg -f concat -i mylist.txt -c copy output.mp4
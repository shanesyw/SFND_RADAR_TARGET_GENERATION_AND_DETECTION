Implementation steps for the 2D CFAR process.
- Following the instructor's skeleton code to understand the logic behind the loops. I am also using the parameter shown in the project introduction video. Filling the loop to iterate through the grid (slide the cell under test across matrix). After summing the noise level (with db2pow conversion), find the threshold by getting the average plus the offset. The cell under test is compared against the threshold and assign a binary result (1 or 0). Originally I was using Max_T like in the video, but eventually realize the result is not good and I have to normalize the grid first in order to get similar result shown in the video.

Selection of Training, Guard cells and offset.
- This is based on the parameters shown by the instructor in the project intro video.

Steps taken to suppress the non-thresholded cells at the edges.
- Basically, if the non-thresholded cells at the edge is not suppressed, there will be a "fuzzy frame" around the edge with values between 0 and 1. Since with the CFAR we are already using a binary output (1 or 0), we can pretty much assign anything that's not 1 as 0.
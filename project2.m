clear all
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 70 m/s  //was 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Velocity resolution 3 m/s
range_resolution = 1; % meter
max_range = 200; % meter
c = 3*10^8; %Speed of light; unit: m/s

%speed of light = 3e8
%% User Defined Range and Velocity of target
% *%TODO* :
% define the target's initial position and velocity. Note : Velocity
% remains contant
d_init = 110; % in meters max. 200m
velocity = -20; % m/s must be -70~70ms


%% FMCW Waveform Generation

% *%TODO* :
%Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.
Bsweep = c / (2*range_resolution); % 150MHz
Tchirp = 5.5*2*max_range/c; % 7.33 us
Slope = Bsweep/Tchirp; % 2.0455 e13

%Operating carrier frequency of Radar 
fc= 77e9;             %carrier freq

                                                          
%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0,Nd*Tchirp,Nr*Nd); %total time for samples
num_of_samples = length(t);

%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));

%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
  
    % *%TODO* :
    %For each time stamp update the Range of the Target for constant velocity. 
    current_range = d_init + t(i)*velocity;
    
    trip_time = 2*current_range/c;
    
    % *%TODO* :
    %For each time sample we need update the transmitted and
    %received signal. 
    Tx(i) = cos(2*pi*(fc*t(i) + (Slope*t(i)^2)/2));
    Rx(i) = cos(2*pi*(fc*(t(i)-trip_time) + (Slope*(t(i)-trip_time)^2)/2));
    
    % *%TODO* :
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix = Tx.*Rx;
end

%% RANGE MEASUREMENT


 % *%TODO* :
%reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
%Range and Doppler FFT respectively.
Mix_ArrayType = reshape(Mix,[Nr,Nd]);

 % *%TODO* :
%run the FFT on the beat signal along the range bins dimension (Nr) and
%normalize.
signal_fft = fft(Mix_ArrayType, Nr);

 % *%TODO* :
% Take the absolute value of FFT output
%L = Nr;
signal_fft2 = abs(signal_fft); % P2 = abs(signal_fft/L);
signal_fft3 = signal_fft2/Nr;
% P1  = P2(1:L/2+1)    

 % *%TODO* :
% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
signal_fft4 = signal_fft3(1:Nr/2);
%P1  = P2(1:L/2+1);    

% Plotting
f = (0:(Nr/2-1));
%f = Fs*(0:(L/2))/L;
%f = 1:Nr;
%plot(f,signal_fft4);
plot(f,signal_fft4);
axis([0 200 0 0.5]);
%plot(signal_fft);
title('Single-Sided Amplitude Spectrum of X(t)');
xlabel('f (Hz)');
ylabel('|spectrum(f)|');


%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);


%% CFAR implementation

%Slide Window through the complete Range Doppler Map


%Slide Window through the complete Range Doppler Map
% *%TODO* :
%Select the number of Training Cells in both the dimensions.
Tr = 10;
Td = 8;

% *%TODO* :
%Select the number of Guard Cells in both dimensions around the Cell under 
%test (CUT) for accurate estimation
Gr = 4;
Gd = 4;

% *%TODO* :
% offset the threshold by SNR value in dB
offset = 1.2; %1.2 is closest to the graph showsn in project description.
% *%TODO* :
%Create a vector to store noise_level for each iteration on training cells
noise_level = zeros(1,1);

% *%TODO* :
%design a loop such that it slides the CUT across range doppler map by
%giving margins at the edges for Training and Guard Cells.
%For every iteration sum the signal level within all the training
%cells. To sum convert the value from logarithmic to linear using db2pow
%function. Average the summed values for all of the training
%cells used. After averaging convert it back to logarithimic using pow2db.
%Further add the offset to it to determine the threshold. Next, compare the
%signal under CUT with this threshold. If the CUT level > threshold assign
%it a value of 1, else equate it to 0.


   % Use RDM[x,y] as the matrix from the output of 2D FFT for implementing
   % CFAR

% Found that I need to normalize the RDM first.
cell_grid = RDM / max(RDM(:));


for i = Tr + Gr + 1 : (Nr/2) - (Tr + Gr)
  for j = Td + Gd + 1 : (Nd) - (Td + Gd)
    %Create a vector to store noise_level for each iteration on training cells
    noise_level = zeros(1, 1);

    for p = i - (Tr + Gr) : i + (Tr + Gr)
      for q = j - (Td + Gd) : j + (Td + Gd)
        if (abs(i - p) > Gr || abs(j - q) > Gd)
          noise_level = noise_level + db2pow(cell_grid(p, q));
        end
      end
    end

    % Calculate threshold from noise average then add the offset
    threshold = pow2db(noise_level / (2 * (Td + Gd + 1) * 2 * (Tr + Gr + 1) - (Gr * Gd) - 1));
    threshold = threshold + offset;

    CUT = cell_grid(i,j);

    if (CUT < threshold)
      cell_grid(i, j) = 0;
    else
      cell_grid(i, j) = 1;
    end

  end
end

% *%TODO* :
% The process above will generate a thresholded block, which is smaller 
%than the Range Doppler Map as the CUT cannot be located at the edges of
%matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 
%cell_grid(cell_grid~=0 & cell_grid~=1) = 0;
cell_grid(cell_grid~=1) = 0;

% *%TODO* :
%display the CFAR output using the Surf function like we did for Range
%Doppler Response output.
figure,surf(doppler_axis, range_axis, cell_grid);
colorbar;
title( '2D CFAR Plot');
xlabel('Doppler');
ylabel('Range');
zlabel('Normalized Amplitude');
sample = importdata('01-AK-1D-LG-LG.dpt'); 
MZ = sample(:,1); %create wavenumbers variable
z = size(MZ,1);

dinfo = dir ('*LG*.dpt'); %read files in current directory with .dpt extension
files = {dinfo.name}; %create a cell array with the names of the files in the directory
a = length(files); %number of files to use in the for loop
Y = zeros(z, a); %preallocation of Y

for i = 1:a %loop extract every second column of each file and append into Y variable
    Y(:,i) = dlmread(files{i},'\t',0,1);
end


plot(MZ,Y)
set(gca, 'XDir','reverse')
title('Mid infrared Spectra of An. gambiae (kisumu) LEGS 1 day old')
xlabel('Wavenumbers (cm -1)') 
ylabel('Absorbance')

Y = Y';
MZ = MZ';


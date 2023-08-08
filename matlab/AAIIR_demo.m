% -----------------------------------------------------------------------%
% AA-IIR WAVEFORM GENERATION
% Authors: L. Gabrielli, P.P. La Pastina, S. D'Angelo - 2021-2022
%
% Matlab implementation of a wavetable oscillator with AA-IIR
% Two waveforms have been implemented: SAW and ESCALATION (See paper)
% -----------------------------------------------------------------------%

pkg load signal

%% DEFINES

Fs = 44100;
% waveform = "ESCALATION";
waveform = "WAVETABLE_SAW";
% waveform = "SAW";
f0 = 1000;


%% BENCHMARK FUNCTION

function runbenchmark(Fs,f0,waveform,stopbdB,cutoff,order,type)
    
    duration = 1.0; % seconds
	time = linspace(0,duration,Fs*duration);
    test_ = zeros(length(time));
    test_(441) = -1.0;

    if strcmp(type,'cheby2')
        stopbF = cutoff;
        [y, iirba] = AAIIROscCheby(time, stopbdB, stopbF, order, Fs, f0, waveform);
    elseif strcmp(type,'butter')
        [y, iirba] = AAIIROscButter(time, cutoff, order, Fs, f0, waveform);
    elseif strcmp(type,'ovs')
        y = OVSTrivial(time, f0, Fs, order, waveform);
    elseif strcmp(type,'trivial')
        y = OVSTrivial(time, f0, Fs, 1, waveform);
    else
        error('wrong method');
    end

    audiowrite("octave_cheby_test.wav", y*0.80, Fs, 'BitsPerSample',32)

    % figure, plot(y(1:100));
    figure, plot(time(1:1100), y(1:1100));
    % figure, plot(I_sums(1:100));
    ylim([-1 1]);
    legend(type);
    % plot(time,10*log10(y));
    [pxx, f] = pwelch(y,4096,[],[],Fs);
    loglog(f, pxx);
    plot(f, log2(pxx));
    grid on;

end


%% DSP METHODS

function [y,iirba] = AAIIROscButter(n, cutoff, order, Fs, f0, waveform)

Fc = cutoff; % [Hz]
Fcrads = 2*pi*Fc / Fs; % [rad/sample]

[z,p,k] = butter(order, Fcrads, 's');
[b,a] = zp2tf(z,p,k);
[r,p,k] = residue(b,a);

L = length(n);
x = (1:L)*f0/Fs;            % vector of x, evenly spaced

y_aa = 0*x;
for o = 1:2:order % poles are in conjg pairs, must take only one for each
    ri = r(o);
    zi = p(o);
    y_aa = y_aa + AA_osc_cplx(x, ri, zi, Fs, waveform);
end

y = y_aa;
iirba = [a, b];

end


function [y,iirba] = AAIIROscCheby(n, stbAtt, stbFreq, order, Fs, f0, waveform)

x = 2 * f0 * mod(n,1/f0) - 1;

Fc = stbFreq;
Fcrads = 2*pi*Fc / Fs;

[z,p,k] = cheby2(order, stbAtt, Fcrads, 's');
[b,a] = zp2tf(z,p,k);
[r,p,k] = residue(b,a);

L = length(x);
x = (1:L)*f0/Fs;

y_aa = 0*x;
for o = 1:2:order % poles are in conjg pairs, must take only one for each
    ri = r(o);
    zi = p(o); % the other is conjugated
    y_aa = y_aa + AA_osc_cplx(x, ri, zi, Fs, waveform);
end

y = y_aa;
iirba = [a; b];

end


function y = OVSTrivial(n, f0, Fs, order, waveform)
    
    duration = max(n);
    nupsmpl = linspace(0,duration,Fs*order*duration);
    trivupsmpl = 2 * f0 * mod(nupsmpl,1/f0) - 1;
    
    if strcmp(waveform, 'SAW')
        yi = trivupsmpl;
    elseif strcmp(waveform, 'ESCALATION')
        % linearly interpolated read
        [~,~,~,wt] = generateEscalationII_w3();
        yi = linint_wt_read(trivupsmpl, wt);
    end
    
    y = decimate(yi,order);
    
end 


function y = linint_wt_read(x, wt)

    N = length(wt);
    dur = length(x);
    
    % x is in range -1:1 but must be in range 1:N to read the wt
    X = (x * (N-1)/2) + (N-1)/2 + 1;
    
    y = 0*x;
    for i = 1:dur
        intx = floor(X(i));
        while intx > N
            intx = intx - N;
        end
        intx_1 = intx+1;
        while intx_1 > N
            intx_1 = intx_1 - N;
        end
        frac = X(i) - intx;
        y(i) = (1-frac)*wt(intx) + frac*wt(intx_1);
    end

end


%% RUN TRIVIAL WAVEFORM GENERATION

% runbenchmark(Fs,f0,waveform,'','','','trivial');

% %% RUN 8x OVERSAMPLING

% runbenchmark(Fs,f0,waveform,'','',8,'ovs');

%% AAIIR: BUTTERWORTH ORDER 2 (mild antialiasing)

stopbF = 0.45 * Fs;
order = 2;
% runbenchmark(Fs,f0,waveform,'',stopbF,order,'butter');

%% AAIIR: CHEBYSHEV ORDER 10 (best antialiasing)

stopbdB = 60;
stopbF = 0.61 * Fs; 
order = 10;
runbenchmark(Fs,f0,waveform,stopbdB,stopbF,order,'cheby2');

while waitforbuttonpress != 1
    continue;
end
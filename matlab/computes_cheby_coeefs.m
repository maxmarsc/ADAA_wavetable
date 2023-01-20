pkg load signal

SAMPLERATES = [44100, 88200];
Fc = 0.61 * 44100;
% Fcrads = 2*pi*Fc / SAMPLERATES; % [rad/sample]
stopbdB = 60;

for i_sr = 1:size(SAMPLERATES, 2)
    sr = SAMPLERATES(i_sr);
    Fcrads = 2*pi*Fc/sr;
    fprintf("=== samplerate %d ===\n",sr);

    for order = 2:2:4
        [z,p,k] = cheby2(order, stopbdB, Fcrads, 's');
        [b,a] = zp2tf(z,p,k);
        [r,p,k] = residue(b,a);

        fprintf("=== order %d ===\n",order);
        for i = 1:2:order
            fprintf("r : %f %fj\n", real(r(i)), imag(r(i)));
            fprintf("p : %f %fj\n", real(p(i)), imag(p(i)));
        endfor
    endfor
endfor
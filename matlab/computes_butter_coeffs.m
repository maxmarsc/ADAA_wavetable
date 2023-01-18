pkg load signal

SAMPLERATES = 88200
Fc = 44100 * 0.45
Fcrads = 2*pi*Fc / SAMPLERATES; % [rad/sample]


for order = 2:2:4
    [z,p,k] = butter(order, Fcrads, 's');
    [b,a] = zp2tf(z,p,k);
    [r,p,k] = residue(b,a);

    fprintf("=== order %d ===\n",order);
    for i = 1:2:order
        fprintf("r : %f %fj\n", real(r(i)), imag(r(i)));
        fprintf("p : %f %fj\n", real(p(i)), imag(p(i)));
    endfor
endfor
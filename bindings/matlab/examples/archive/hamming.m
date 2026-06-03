function value = hamming(ratio)
    value = double(ratio<=0.5).*(0.53836 + 0.46164*cos(2*pi*ratio));
end
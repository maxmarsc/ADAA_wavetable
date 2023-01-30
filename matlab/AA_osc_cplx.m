function [y] = AA_osc_cplx(x, resid, polo, fs, wave)
  
  % waveform info
  if strcmp(wave,'ESCALATION')
      [X,m,q,~] = generateEscalationII_w3();
  elseif strcmp(wave,'WAVETABLE_SAW')
      [X,m,q, ~] = generateWavetableSaw();
  else
      % SAW
      X = [0,1];
      m = [2];
      q = [-1];
      
  endif

  %%%
  % X | list[scalar] E [0;1] : normalized sample position relative to the waveform length
  % k | scalar : number of samples inside the waveform
  %
  % x | list[scalar] : of phase offsets
  % x_red | scalar : phase offset modulo T (which is 1 but whatever)
  %
  % j | scalar : idx of an X sample
  % j_red | scalar : j kinda modulo k  
  %
  % y | list[scalar] : result
  %
  %%%

  
  T = X(end); % period
  k = length(m); % nr segments
  
  % compute diff
  for j = 1:k - 1
    m_diff(j) = m(j + 1) - m(j);
    q_diff(j) = q(j + 1) - q(j);  
  end
  m_diff(k) = m(1) - m(k);
  q_diff(k) = q(1) - q(k) - m(1) * T;

  % filter
  % beta is alpha inside the paper
  B = resid;
  beta = polo;
  expbeta = exp(beta);
  
  % initial conditions
  x_vz1 = 0;
  y_hat_vz1 = 0;
  x_diff_vz1 = 0;
  
  % index j corresponding to initial sample x_vz1  
  x_red = mod(x_vz1, T);
  j_red = binary_search_down(X, x_red, 1, length(X));
  j = k * floor(x_vz1/T) + j_red - 1;   % seems to be an index, not a complex
  % j seem to be the index of x, the phase offset
  
  % Process
  for n = 2:length(x)
    
    % x_vz1 is the precedent value of x
    x_diff = x(n) - x_vz1;
    j_vz1 = j;
            
    if ((x_diff >= 0 && x_diff_vz1 >= 0) || (x_diff < 0 && x_diff_vz1 <= 0))
      % If on the same slope than previous iteration
      j_vz1 = j + sign(x_red - X(j_red));
      % +1 or -1 or +0 depending on the sign of x_red - X(j_red)
    end
    
    x_red = mod(x(n), T);
    
    if (x_diff >= 0)
      j_red = binary_search_down(X, x_red, 1, length(X));
      j = k * floor(x(n)/T) + j_red - 1;
      j_min = j_vz1;
      j_max = j;
    else
      j_red = binary_search_up(X, x_red, 1, length(X));
      j = k * floor(x(n)/T) + j_red - 1;
      j_min = j;
      j_max = j_vz1;
    end
    
    j_min_bar = mod_bar(j_min, k);
    j_max_p_bar = mod_bar(j_max + 1, k);
    
    if (x_diff >= 0)      
      I = expbeta \
          * (m(j_min_bar) * x_diff + beta * (m(j_min_bar) * (x_vz1 - T * floor((j_min - 1)/k)) + q(j_min_bar)))\
          - m(j_max_p_bar) * x_diff \
          - beta * (m(j_max_p_bar) * (x(n) - T * floor(j_max/k)) + q(j_max_p_bar));
    else
      I = expbeta \
          * (m(j_max_p_bar) * x_diff + beta * (m(j_max_p_bar) * (x_vz1 - T * floor(j_max/k)) + q(j_max_p_bar))) \
          - m(j_min_bar) * x_diff \
          - beta * (m(j_min_bar) * (x(n) - T * floor((j_min - 1)/k)) + q(j_min_bar));
    end
      
    I_sum = 0;
    s_parts = zeros(1, j_max - j_min);
    for l = j_min:j_max
      l_bar = mod_bar(l, k);
      I_sum = I_sum \
          + exp(beta * (x(n) - X(l_bar + 1) - T * floor((l - 1)/k))/x_diff) \
              * (beta * q_diff(l_bar) + m_diff(l_bar) * (x_diff + beta * X(l_bar + 1)));
    end

    I = (I + sign(x_diff) * I_sum)/beta^2;
    
    % See formula n°10
    y_hat = expbeta * y_hat_vz1 + 2 * B * I;
    % We take the real part of y
    y(n) = real(y_hat);
      
    x_vz1 = x(n);
    y_hat_vz1 = y_hat;
    x_diff_vz1 = x_diff;
    
  end
   
endfunction

% i tel que  x_i < x_0 < x_(i+1) && j_min <= i <= j_max
%            x_i_m1 < x_m1 < x_i
function y = binary_search_down(x, x0, j_min, j_max)
    % index of last number in ordered vec x <= x0, among those between j_min, j_max. 
    % if x0 < x(1), return 0.

    if (x0 < x(1))
      y = 0;
    elseif (x0 >= x(j_max))
      y = j_max;
    else
      i_mid = floor((j_min + j_max)/2);
      if (x0 < x(i_mid))
        j_max = i_mid;
      elseif (x0 == x(i_mid))
        y = i_mid;
        return
      else
        j_min = i_mid;
      end
      if (j_max - j_min > 1)
        y = binary_search_down(x, x0, j_min, j_max);
      else
        y = j_min;
      end
    end
   
endfunction


function y = binary_search_up(x, x0, j_min, j_max)
  
    if (x0 > x(end))
      y = length(x) + 1;
    elseif (x0 <= x(1))
      y = 1;
    else
      i_mid = floor((j_min + j_max)/2);
      if (x0 < x(i_mid))
        j_max = i_mid;
      elseif (x0 == x(i_mid))
        y = i_mid;
        return
      else
        j_min = i_mid;
      end
      if (j_max - j_min > 1)
        y = binary_search_up(x, x0, j_min, j_max);
      else
        y = j_max;
      end
    end
   
endfunction

% The weird mod that doesn't go to zero
% defined after formula (18)
function y = mod_bar(x, k)
  % return mod(x, k), if not 0 else return k
  m = mod(x, k);
  y = m + k * (1 - sign(m));
endfunction

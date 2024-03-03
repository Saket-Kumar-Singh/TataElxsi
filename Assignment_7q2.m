N = 16;
y_ = func(16);
y1_ = y(1:17);
y2_ = y(18:34);
x = linspace(0, 1, 17);
er = [];
n = [];

for i = 1:15
N = N+16;
y = func(N);
y1 = y(1:N+1);
y2 = y(N+2:2*N+2);
x = linspace(0, 1, N+1);


err = 0;
p = 1;
for k = 1:N/2
    err = err + (y1(p, 1) - y1_(k, 1))^2;
    p = p+2;
end    
er = [er, err];
n = [n, N];


N = N + 16;
y1_ = y1;
y2_ = y2;

end


scatter( n, er);
hold on
plot(n, er);
hold off;

function y = func(N)
    A = zeros(2*N+2, 2*N+2);
    b = zeros(2*N+2, 1);
    A(1, 1) = 1;
    A(N+1, N+1) = 1;
    A(N+2, N+2) = 1;
    A(2*N + 2, 2*N + 2) = 1;
    h = 1/N;
    for i = 2:N
        x = i/N;        
        A(i, i+1) = 1*((1 + x^2)/h^2);
        A(i, i) = (-2*(1 + x^2)/h^2 - 1/h + 10);
        A(i, i-1) = ((1 + x^2)/h^2 + 1/h);
        A(i, i + N + 1) = -1;
        b(i) = 1;
    end    
    for i = N+3:2*N+1
        x = (i - N - 1)/N;
        A(i, i+1) = 1/h^2;
        A(i, i) = (-2/h^2 - 1/h + 10*(x+1));
        A(i, i-1) = (1/h^2 + 1/h);
        A(i , i - N -1) = -5;
    end    
    y = A\b;
end
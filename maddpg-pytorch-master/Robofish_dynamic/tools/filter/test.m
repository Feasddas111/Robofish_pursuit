clf;
wp1=0.5;wp2=3.5;
ws1=1.9;ws2=2.1;
rp=0.1; % ͨ���������˥��Ϊ0.1dB
rs=100; % ���˥��Ϊ60dB
fs=10; % ����Ƶ��Ϊ400Hz
wp=[wp1,wp2]/(fs/2);% ͨ����ֹƵ��/Hz
ws=[ws1,ws2]/(fs/2);% �����ֹƵ��/Hz
% ������˹�˲���
[b_N,b_wc]=buttord(wp,ws,rp,rs,'z');
b_N = 5;
b_wc = [0.05, 0.95];
[b_b,b_a]=butter(b_N,b_wc,'stop');
[b_hw,b_w]=freqz(b_b,b_a,512); % ����Ƶ����Ӧ����
A1 = b_w/(2*pi)*fs;
plot(A1,abs(b_hw));


% ԭʼ����
N=128;   %����Ƶ�ʺ����ݵ���
n=0:N-1;t=n/fs;   %ʱ������
f=n*fs/N;    %Ƶ������
x=2*sin(2*pi*1.5*t); %�ź�
% x = temp(:, 6);
y=fft(x,N);    %���źŽ��п���Fourier�任
mag=abs(y);     %���Fourier�任������
subplot(2,2,1),plot(f,mag);   %�����Ƶ�ʱ仯�����
xlabel('Ƶ��/Hz');
ylabel('���');title('N=128');grid on;
% subplot(2,2,2),plot(f(1:N/2),mag(1:N/2)); %���NyquistƵ��֮ǰ��Ƶ�ʱ仯�����
% xlabel('Ƶ��/Hz');
% ylabel('���');title('N=128');grid on;
subplot(2,2,2)
plot(x)
% �˲�����
xf=filter(b_b,b_a, x); % �����ź�ͨ��butter�˲�������ź�yn;
yf=fft(xf,N); % ���ź���M��FFT�任
mag=abs(yf);     %���Fourier�任������
subplot(2,2,3),plot(f,mag);   %�����Ƶ�ʱ仯�����
xlabel('Ƶ��/Hz');
ylabel('���');title('N=128');grid on;
% subplot(2,2,2),plot(f(1:N/2),mag(1:N/2)); %���NyquistƵ��֮ǰ��Ƶ�ʱ仯�����
% xlabel('Ƶ��/Hz');
% ylabel('���');title('N=128');grid on;
subplot(2,2,4)
plot(xf)
hold on
% �Լ�ʵ�ֵ��˲�
prev_in_data = zeros(11,1);
prev_out_data = zeros(10,1);
xff = x;
for jj = 1:length(x)
    % Ų���������
    for ii = length(prev_in_data)-1:-1:1
        prev_in_data(ii+1) = prev_in_data(ii);
    end
    prev_in_data(1) = x(jj);
    out = -b_a(2:end)*prev_out_data + b_b*prev_in_data;
    % Ų���������
    for ii = length(prev_out_data)-1:-1:1
        prev_out_data(ii+1) = prev_out_data(ii);
    end
    prev_out_data(1) = out;
    xff = out;
end
plot(xff)
hold on

% %% IIR�ݲ������
% % Ŀ�ģ����һ���ݲ��������50��1.5Hz����,����Ƶ��Ϊ400Hz���˲�����
% % ��Ҫ��ͨ�����˥��Ϊ0.1dB�������С˥��Ϊ60dB��
% clc;
% clear;close all;
% wp1=48.5;wp2=51.5;
% ws1=49.9;ws2=50.1;
% rp=0.1; % ͨ���������˥��Ϊ0.1dB
% rs=100; % ���˥��Ϊ60dB
% fs=400; % ����Ƶ��Ϊ400Hz
% wp=[wp1,wp2]/(fs/2);% ͨ����ֹƵ��/Hz
% ws=[ws1,ws2]/(fs/2);% �����ֹƵ��/Hz
% 
% %% -------------------------------�˲������----------------------------------%%
% 
% % ������˹�˲���
% [b_N,b_wc]=buttord(wp,ws,rp,rs,'z');
% [b_b,b_a]=butter(b_N,b_wc,'stop');
% [b_hw,b_w]=freqz(b_b,b_a,512); % ����Ƶ����Ӧ����
% 
% % �б�ѩ��I���˲���
% [c1_N,c1_wc]=cheb1ord(wp,ws,rp,rs,'z');
% [c1_b,c1_a]=cheby1(c1_N,rp,c1_wc,'stop');
% [c1_hw,c1_w]=freqz(c1_b,c1_a,512);% ����Ƶ����Ӧ����
% 
% %�б�ѩ��II���˲���
% [c2_N,c2_wc]=cheb2ord(wp,ws,rp,rs,'z');
% [c2_b,c2_a]=cheby2(c2_N,rs,c2_wc,'stop');
% [c2_hw,c2_w]=freqz(c2_b,c2_a,512);% ����Ƶ����Ӧ����
% 
% % ��Բ�˲���
% [d_N,d_wc]=ellipord(wp,ws,rp,rs,'z');
% [d_b,d_a]=ellip(d_N,rp,rs,d_wc,'stop');
% [d_hw,d_w]=freqz(d_b,d_a,512); % ����Ƶ����Ӧ����
% 
% %% -------------------- �����˲����ķ�Ƶ��Ӧ�Աȷ���---------------------%%
% A1 = b_w/(2*pi)*fs;
% A2 = c1_w/(2*pi)*fs;
% A3 = c2_w/(2*pi)*fs;
% A4 = d_w/(2*pi)*fs;
% 
% figure; % ��ͼ
% subplot(2,1,1);
% plot(A1,abs(b_hw),A2,abs(c1_hw),A3,abs(c2_hw),A4,abs(d_hw));grid;
% title('H(ejw)'); xlabel('Ƶ�ʣ�Hz'); ylabel('Ƶ����Ӧ����');
% legend('butter','cheby1','cheby2','ellip');
% % axis([47,53,0,1.1]);% ����������������ķ�Χ
% subplot(2,1,2);
% plot(A1,20*log10(abs(b_hw))/max(abs(b_hw)),A2,20*log10(abs(c1_hw))/max(abs(c1_hw)),A3,20*log10(abs(c2_hw))/max(abs(c2_hw)),A4,20*log10(abs(d_hw))/max(abs(d_hw)));
% title('��ĺ���'); xlabel('Ƶ��/Hz');ylabel('��ֵ/dB');
% legend('butter','cheby1','cheby2','ellip');
% % axis([47,53,-130,10]); % ����������������ķ�Χ
% grid on;
% 
% %% ---------------------------- �����ź�-----------------------------%%
% 
% M=2038400;
% n=0:M-1;
% T=1/fs;
% xn=sin(2*pi*48*n*T)+sin(2*pi*48.5*n*T)+sin(2*pi*49*n*T)+sin(2*pi*50*n*T)+sin(2*pi*51*n*T)+sin(2*pi*51.5*n*T)+sin(2*pi*52*n*T); % �źŵ���
% %--------------------------- FFT�����ź�Ƶ��-------------------------------%
% xk=fft(xn,M); % ���ź���len��FFT�任
% yn_b=filter(b_b,b_a,xn); % �����ź�ͨ��butter�˲�������ź�yn;
% yn_c1=filter(c1_b,c1_a,xn); % �����ź�ͨ��cheby1�˲�������ź�yn;
% yn_c2=filter(c2_b,c2_a,xn); % �����ź�ͨ��cheby2�˲�������ź�yn;
% yn_d=filter(d_b,d_a,xn); % �����ź�ͨ��ellip�˲�������ź�yn;
% k=2*(0:M-1)/M;
% 
% yk_b=fft(yn_b,M); % ���ź���M��FFT�任
% yk_c1=fft(yn_c1,M); % ���ź���M��FFT�任
% yk_c2=fft(yn_c2,M); % ���ź���M��FFT�任
% yk_d=fft(yn_d,M);
% % 
% % yk_b_T=Tfft(yn_b,M); % ���ź���M��FFT�任
% % yk_c1_T=Tfft(yn_c1,M); % ���ź���M��FFT�任
% % yk_c2_T=Tfft(yn_c2,M); % ���ź���M��FFT�任
% % yk_d_T=Tfft(yn_d,M); % ���ź���M��FFT�任
% yk_b_T=fft(yn_b,M); % ���ź���M��FFT�任
% yk_c1_T=fft(yn_c1,M); % ���ź���M��FFT�任
% yk_c2_T=fft(yn_c2,M); % ���ź���M��FFT�任
% yk_d_T=fft(yn_d,M); % ���ź���M��FFT�任
% %% -------------------------------������ʾ----------------------------------%%
% kfs = n*fs/M;
% % ԭ�ź�
% figure(2); % ��ͼ
% subplot(2,1,1);
% plot(n,xn,'blue');
% title('ԭ�źŲ���ͼ'); xlabel('f/Hz');ylabel('����');
% axis([0,4000,-8,8]); % ����������������ķ�Χ
% grid on;
% subplot(2,1,2);
% plot(kfs/2,abs(xk),'blue'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=k*fs/2
% title('�˲�ǰ�����ź�x(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');%
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(xk))]);% ����������������ķ�Χ
% grid on;
% 
% % ������˹�˲���
% figure; % ��ͼ
% subplot(3,1,1);
% plot(n,yn_b,'r');
% title('butter�˲�����ͼ'); xlabel('f/Hz');ylabel('����');
% axis([0,4000,-8,8]); % ����������������ķ�Χ
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_b),'b'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('fft�任');
% title('butter�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% %axis([40,60,0,max(abs(yk_b))]); % ����������������ķ�Χ
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_b_T),'r'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('T*fft�任');
% title('butter�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_b_T))]); % ����������������ķ�Χ
% grid on;
% 
% % �б�ѩ��I���˲���
% figure; % ��ͼ
% subplot(3,1,1);
% plot(n,yn_c1,'m');
% title('cheby1�˲�����ͼ'); xlabel('f/Hz');ylabel('����');
% axis([0,4000,-8,8]); % ����������������ķ�Χ
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_c1),'m'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('fft�任');
% title('cheby1�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c1))]);% ����������������ķ�Χ
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_c1_T),'m'); % k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('T*fft�任');
% title('cheby1�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c1_T))]);% ����������������ķ�Χ
% grid on;
% 
% % �б�ѩ��II���˲���
% figure; % ��ͼ
% subplot(3,1,1);
% plot(n,yn_c2,'g');
% title('cheby2�˲�����ͼ'); xlabel('f/Hz');ylabel('����');
% axis([0,4000,-8,8]); % ����������������ķ�Χ
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_c2),'g');% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('fft�任');
% title('cheby2�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c2))]);%����������������ķ�Χ
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_c2_T),'g');%����ɢͼ k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('T*fft�任');
% title('cheby2�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_c2_T))]);% ����������������ķ�Χ
% grid on;
% 
% % 3.4 ��Բ�˲���
% figure; % ��ͼ
% subplot(3,1,1);
% plot(n,yn_d);
% title('ellip�˲�����ͼ'); xlabel('f/Hz');ylabel('����');
% axis([0,4000,-8,8]); % ����������������ķ�Χ
% grid on;
% subplot(3,1,2);
% plot(kfs/2,abs(yk_d));% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('fft�任');
% title('ellip�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_d))]);% ����������������ķ�Χ
% grid on;
% subplot(3,1,3);
% plot(kfs/2,abs(yk_d_T));% k=w/pi,w=mo_wT=mo_w/fs,mo_w=2pif;�õ�f=kfs/2
% legend('T*fft�任');
% title('ellip�˲�������ź�y(n)Ƶ������'); xlabel('f/Hz');ylabel('H(ejw)');
% % axis([40,60,0,6000]);
% % axis([40,60,0,max(abs(yk_d_T))]);% ����������������ķ�Χ
% grid on;
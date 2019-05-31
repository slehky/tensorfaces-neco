function dim = DimSelector(out);
% DimSelector(out);
%  Selecting among three-mode principal components models of different types and complexity: a numerical convex hull based method
%  Eva Ceulemans and Henk A. L. Kiers. British journal of mathematical and statistical psychology, 2006, 59, 133-150.
%
% INPUT
% 
%  X   (frontal slices next to each other)
%  out % from tuckruns; contains in columns: r1,r2,r3 and fit
% uses ed.m ssq.m LineCon.m cpfunc.m
%
% Written by Urbano Lorenzo-Seva, Rovira i Virgili University (Last update: November 22, 2007)

%P,Q,R : maximum number of dimensions 
P=max(out(:,1));
Q=max(out(:,2));
R=max(out(:,3));

% completing out by r1+r2+r3
out(:,5)=out(:,1:3)*ones(3,1);

%Sorting by total number of components (5th column of out)
[ss,si]=sort(out(:,5));
out=out(si,:);
%writescr(out,'10.4')
%pause
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Convex hull based model selection procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Step 2 Retain only the best-fitting solutions
[tmp1,tmp2]=size(out);
fpi=min(out(:,5));
for i=1:tmp1,
    if out(i,5) ~= max(fpi),
        fpi=[fpi; out(i,5)];
    end;
end;

[tmp3,tmp2]=size(fpi);
out_best =zeros (tmp3,5);
for i=1:tmp3,
    for j=1:tmp1,
        if fpi(i) == out(j,5),
            if out_best(i,4) < out(j,4),
               out_best(i,:) = out(j,:); 
            end;
        end;
    end;
end;

%Step 4 Clean for f
out_best_best =out_best(1,:);
for i=2:tmp3,
    if max(out_best_best(:,4))< out_best(i,4),
        out_best_best = [out_best_best; out_best(i,:)];
    end;
end;

%Steps 5 i 6 Testing triplets of adjacent solutions
[tmp4,tmp2]=size(out_best_best);
out_best_best_best=out_best_best(1,:);

i=2;
while i < tmp4-1,
        f1 =out_best_best(i-1,4);
        f2 =out_best_best(i  ,4);
        f3 =out_best_best(i+1,4);
        fp1=out_best_best(i-1,5);
        fp2=out_best_best(i  ,5);
        fp3=out_best_best(i+1,5);
        if LineCon(f1,f2,f3,fp1,fp2,fp3) == 0,
           out_best_best = [out_best_best(1:i-1,:); out_best_best(i+1:tmp4,:)];
           [tmp4,tmp2]=size(out_best_best);
           i=1; 
        end;
        i = i+1;
end;

%Step 7 Compute St
[tmp4,tmp2]=size(out_best_best);
st = zeros(tmp4,1);
for i=2:tmp4-1,
    fi    = out_best_best(i,  4);
    fi_p  = out_best_best(i-1,4);
    fi_n  = out_best_best(i+1,4);
    fpi   = out_best_best(i,  5);
    fpi_p = out_best_best(i-1,5);
    fpi_n = out_best_best(i+1,5);
    st(i) = ((fi - fi_p) / (fpi - fpi_p)) / ((fi_n - fi) / (fpi_n - fpi));
end;

out_st = [out_best_best st];

%Step 8 Select the solution with highest st-value
[st_tmp,imax]= max (out_st(:,6));
st_max = out_st(imax,:);

%Step 9 Look for equivalent solutions
sol_equi=[];e=0;
for i=1:tmp1,
    if st_max(5) == out(i,5) & st_max(4) ~= out(i,4),
        sol_equi = [sol_equi; out(i,:)];
        e=e+1;
    end;
end;

%REPORT FOR THE OUTPUT

%Table 

disp(' ');
disp('Table. Goodness-of-fit values f, total number of components Sc, and scree test');
disp('       values st of the solutions on the higher boundary of the convex hull');
disp(' ');
disp('     Model  Complexity    f         Sc      st');
disp(' ');
for i=1:tmp4,
    if out_st(i,1) == -1,
       buff  =sprintf('     CP        %2.0f',out_st(i,3));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The  hull neuristic indicates the selection of the CP model of complexity (%2.0f).',out_st(i,3));;end;
    end;
    if out_st(i,1) == 2 & out_st(i,3) == 0,
       buff  =sprintf('     T1A       %2.0f',out_st(i,1));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull neuristic indicates the selection of the T1A model of complexity (%2.0f).',out_st(i,1));;end;
    end;
    if out_st(i,1) == 0 & out_st(i,3) == 0,
       buff  =sprintf('     T1B       %2.0f',out_st(i,2));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T1B model of complexity (%2.0f).',out_st(i,2));;end;
    end;
    if out_st(i,1) == 0 & out_st(i,2) == 0,
       buff  =sprintf('     T1C       %2.0f',out_st(i,3));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T1C model of complexity (%2.0f).',out_st(i,3));;end;
    end;
    if out_st(i,1) ~= 0 & out_st(i,2) ~= 0 & out_st(i,3) == 0,
       buff  =sprintf('     T2AB    (%2.0f,%2.0f)',out_st(i,1),out_st(i,2));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T2AB model of complexity (%2.0f,%2.0f).',out_st(i,1),out_st(i,2));;end;
    end;
   if out_st(i,1) ~= 0 & out_st(i,2) == 0 & out_st(i,3) ~= 0,
       buff  =sprintf('     T2AC    (%2.0f,%2.0f)',out_st(i,1),out_st(i,3));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T2AC model of complexity (%2.0f,%2.0f).',out_st(i,1),out_st(i,3));;end;
    end;
    if out_st(i,1) == 0 & out_st(i,2) ~= 0 & out_st(i,3) ~= 0,
       buff  =sprintf('     T2BC    (%2.0f,%2.0f)',out_st(i,2),out_st(i,3));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T2BC model of complexity (%2.0f,%2.0f).',out_st(i,2),out_st(i,3));;end;
    end;
    if out_st(i,1) > 0 & out_st(i,2) > 0 & out_st(i,3) > 0,
       buff  =sprintf('     T3    (%2.0f,%2.0f,%2.0f)',out_st(i,1),out_st(i,2),out_st(i,3));
       if i == 1,
           buff2 =sprintf('   %5.4f %7d      -',out_st(i,4), out_st(i,5));
       elseif i == tmp4,
           buff2 =sprintf('   %5.4f %7d      -',out_st(i,4), out_st(i,5));
       else
           buff2 =sprintf('   %5.4f %7d    %5.2f',out_st(i,4), out_st(i,5), out_st(i,6));
       end;
       if i == imax, buff3 = sprintf('The hull heuristic indicates the selection of the T3 model of complexity (%2.0f,%2.0f,%2.0f).',out_st(i,1),out_st(i,2),out_st(i,3));;end;
    end;

    disp([buff buff2]);
    
end;
disp(' ');
disp(buff3);
disp(' ');

dim = out_st(imax,3);

return  % next is suppressed

if (e>0),
    disp('Other solutions are equivalent to the solution selected. They could also be');
    disp('selected and reported. The equivalent solutions are the following:');
disp(' ');
disp('     Model  Complexity    f         Sc      ');
disp(' ');
for i=1:e,
    if sol_equi(i,1) == -1,
       buff  =sprintf('     CP        %2.0f',sol_equi(i,3));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) == 2 & sol_equi(i,3) == 0,
       buff  =sprintf('     T1A       %2.0f',sol_equi(i,1));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d     ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d     ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('       %5.4f %7d   ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) == 0 & sol_equi(i,3) == 0,
       buff  =sprintf('     T1B       %2.0f',sol_equi(i,2));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('       %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) == 0 & sol_equi(i,2) == 0,
       buff  =sprintf('     T1C       %2.0f',sol_equi(i,3));
       if i == 1,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('       %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('       %5.4f %7d   ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) ~= 0 & sol_equi(i,2) ~= 0 & sol_equi(i,3) == 0,
       buff  =sprintf('     T2AB    (%2.0f,%2.0f)',sol_equi(i,1),sol_equi(i,2));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
   if sol_equi(i,1) ~= 0 & sol_equi(i,2) == 0 & sol_equi(i,3) ~= 0,
       buff  =sprintf('     T2AC    (%2.0f,%2.0f)',sol_equi(i,1),sol_equi(i,3));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) == 0 & sol_equi(i,2) ~= 0 & sol_equi(i,3) ~= 0,
       buff  =sprintf('     T2BC    (%2.0f,%2.0f)',sol_equi(i,2),sol_equi(i,3));
       if i == 1,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('    %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('    %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;
    if sol_equi(i,1) > 0 & sol_equi(i,2) > 0 & sol_equi(i,3) > 0,
       buff  =sprintf('     T3    (%2.0f,%2.0f,%2.0f)',sol_equi(i,1),sol_equi(i,2),sol_equi(i,3));
       if i == 1,
           buff2 =sprintf('   %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       elseif i == tmp4,
           buff2 =sprintf('   %5.4f %7d      ',sol_equi(i,4), sol_equi(i,5));
       else
           buff2 =sprintf('   %5.4f %7d    ',sol_equi(i,4), sol_equi(i,5));
       end;
    end;

    disp([buff buff2]);
    
end;
disp(' ');    
end;    
disp(' ');

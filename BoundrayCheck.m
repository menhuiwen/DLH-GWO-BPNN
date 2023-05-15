
%狼群越界调整函数
function x=BoundrayCheck(x,ub,lb,dim)
for i=1:size(x,1)
    for j=1:dim
        if x(i,j)>ub(j)
            x(i,j)=ub(j);
        end
        if x(i,j)<lb(j)
            x(i,j)=lb(j);
        end
    end
end

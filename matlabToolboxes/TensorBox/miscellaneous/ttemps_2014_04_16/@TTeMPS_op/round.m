function A = round(A, tol )
    C = cell(1, A.order);
    for i = 1:A.order
        C{i} = reshape(A.U{i}, [A.rank(i), A.size_col(i)*A.size_row(i), A.rank(i+1)]);
    end
    X = TTeMPS( C );
    X = round(X, tol);
    for i = 1:A.order
        A.U{i} = reshape(X.U{i}, [X.rank(i), A.size_col(i), A.size_row(i), X.rank(i+1)]);
    end

    A = update_properties(A);

end
function Cell2Mat=Cell2Mat4real(initialCell)
    max_len = max(cellfun(@numel, initialCell));
    Cell2Mat=zeros(length(initialCell),max_len);
    for i=1:size(initialCell,2)
        raw_vector=cell2mat(initialCell(i));
        Cell2Mat(i,1:length(raw_vector))=raw_vector;
    end
end
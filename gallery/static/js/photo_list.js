function createSubCategory(parentId, csrfToken) {
    const subName = prompt('새로운 세부 폴더 이름을 입력하세요:');
    if (!subName) return;

    fetch(`/gallery/categories/add/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ 
            name: subName, 
            parent_id: parentId 
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            location.reload();
        } else {
            alert('폴더 생성에 실패했습니다: ' + (data.error || '알 수 없는 오류'));
        }
    })
    .catch(err => {
        console.error('Error:', err);
        alert('서버와의 통신 중 오류가 발생했습니다.');
    });
}
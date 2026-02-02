function toggleBookmark(photoId, csrfToken) {
    const icon = document.getElementById('bookmark-icon');
    const isBookmarked = icon.innerText.includes('취소');
    
    const url = isBookmarked ? `/gallery/bookmarks/${photoId}/` : `/gallery/bookmarks/add/`;
    const method = isBookmarked ? 'DELETE' : 'POST';
    const body = isBookmarked ? null : JSON.stringify({ photoId: photoId });

    fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: body
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            if (isBookmarked) {
                icon.innerText = '☆ 북마크 추가';
                icon.style.color = '#888';
            } else {
                icon.innerText = '⭐ 북마크 취소';
                icon.style.color = 'orange';
            }
        } else {
            alert('북마크 처리에 실패했습니다.');
        }
    })
    .catch(err => console.error('Error:', err));
}

function saveMemo(photoId, csrfToken) {
    const content = document.getElementById('memo-content').value;
    fetch(`/gallery/memos/photos/${photoId}/`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ content: content })
    })
    .then(res => res.json())
    .then(data => {
        if(data.success) alert('메모가 저장되었습니다!');
    });
}

function moveToTrash(photoId, csrfToken) {
    if (!confirm('이 사진을 휴지통으로 보내시겠습니까?')) return;
    fetch(`/gallery/trash/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({ photoId: photoId })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert('휴지통으로 이동되었습니다.');
            location.href = '/gallery/photos/';
        } else {
            alert('이동 실패: ' + (data.error || '알 수 없는 오류'));
        }
    });
}
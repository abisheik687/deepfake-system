export function SkeletonRow({ cols = 5 }) {
    return (
        <tr>
            {Array.from({ length: cols }).map((_, i) => (
                <td key={i} className='px-4 py-3'>
                    <div className='h-4 rounded animate-pulse'
                        style={{ background: '#1a2f4a', width: `${60 + (i % 3) * 15}%` }} />
                </td>
            ))}
        </tr>
    );
}
